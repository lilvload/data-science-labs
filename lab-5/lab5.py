import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def file_parsing(File_name):
    d = pd.read_excel(File_name)
    df = d[["Купівля", "Продаж", "КурсНбу"]].astype(float).reset_index(drop=True)
    return df


def cluster_numeric(df):
    X = df[["Купівля", "Продаж", "КурсНбу"]].values
    Xs = StandardScaler().fit_transform(X)
    sse = []
    sil = []
    Ks = range(2, 8)
    for k in Ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=9)
        labels = km.fit_predict(Xs)
        sse.append(km.inertia_)
        sil.append(silhouette_score(Xs, labels))
    best_k = list(Ks)[int(np.argmax(sil))]
    km = KMeans(n_clusters=best_k, n_init=10, random_state=9)
    labels = km.fit_predict(Xs)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(list(Ks), sse, "o-", color="#1f77b4")
    ax[0].set_xlabel("Кількість кластерів k")
    ax[0].set_ylabel("SSE (інерція)")
    ax[0].set_title("Метод ліктя")
    ax[0].grid(alpha=0.3)
    ax[1].plot(list(Ks), sil, "o-", color="#d62728")
    ax[1].set_xlabel("Кількість кластерів k")
    ax[1].set_ylabel("Silhouette score")
    ax[1].set_title("Силуетний коефіцієнт")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig_numeric_metrics.png", dpi=150)
    plt.close()

    fig = plt.figure(figsize=(8, 6))
    axp = fig.add_subplot(111, projection="3d")
    sc = axp.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="viridis", s=18)
    axp.set_xlabel("Купівля")
    axp.set_ylabel("Продаж")
    axp.set_zlabel("КурсНбу")
    axp.set_title(f"K-means кластеризація даних Ощадбанк (k={best_k})")
    fig.colorbar(sc, ax=axp, shrink=0.6, label="Кластер")
    plt.tight_layout()
    plt.savefig("figures/fig_numeric_clusters.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.scatter(np.arange(len(labels)), df["Продаж"].values, c=labels, cmap="viridis", s=18)
    plt.xlabel("Номер спостереження")
    plt.ylabel("Продаж USD")
    plt.title("Розподіл кластерів у часі (показник Продаж)")
    plt.grid(alpha=0.3)
    plt.colorbar(label="Кластер")
    plt.tight_layout()
    plt.savefig("figures/fig_numeric_time.png", dpi=150)
    plt.close()

    return best_k, sil, list(Ks), labels


def cluster_image_color(path, k=4):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    data = img.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=10, random_state=9)
    labels = km.fit_predict(data)
    centers = km.cluster_centers_.astype(np.uint8)
    seg = centers[labels].reshape(h, w, 3)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].imshow(img)
    ax[0].set_title("Вихідне зображення")
    ax[0].axis("off")
    ax[1].imshow(seg)
    ax[1].set_title(f"Сегментація K-means (k={k})")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig("figures/fig_image_segmentation.png", dpi=150)
    plt.close()

    seg_imgs = {k: seg}
    for kk in (5,):
        km2 = KMeans(n_clusters=kk, n_init=10, random_state=9)
        lab2 = km2.fit_predict(data)
        cen2 = km2.cluster_centers_.astype(np.uint8)
        seg_imgs[kk] = cen2[lab2].reshape(h, w, 3)
    fig, ax = plt.subplots(1, len(seg_imgs), figsize=(6 * len(seg_imgs), 4.5))
    for i, kk in enumerate(sorted(seg_imgs)):
        ax[i].imshow(seg_imgs[kk])
        ax[i].set_title(f"k = {kk}")
        ax[i].axis("off")
    plt.suptitle("Вплив кількості кластерів на сегментацію")
    plt.tight_layout()
    plt.savefig("figures/fig_image_k_compare.png", dpi=150)
    plt.close()

    masks = []
    for c in range(k):
        mask = (labels.reshape(h, w) == c).astype(np.uint8) * 255
        masks.append(mask)
    fig, ax = plt.subplots(1, k, figsize=(3.2 * k, 3.4))
    for c in range(k):
        rgb = np.tile(masks[c][:, :, None], (1, 1, 3))
        tint = (rgb / 255.0 * centers[c]).astype(np.uint8)
        ax[c].imshow(tint)
        col = centers[c]
        ax[c].set_title(f"Кластер {c}\nRGB=({col[0]},{col[1]},{col[2]})")
        ax[c].axis("off")
    plt.tight_layout()
    plt.savefig("figures/fig_image_masks.png", dpi=150)
    plt.close()
    return centers


def count_objects(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)[1].astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    count = 0
    out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    for m in range(2, markers.max() + 1):
        ys, xs = np.where(markers == m)
        if len(xs) < 200:
            continue
        count += 1
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.circle(out, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(out, str(count), (cx - 10, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Вихідне зображення")
    ax[0].axis("off")
    ax[1].imshow(dist, cmap="jet")
    ax[1].set_title("Карта відстаней (distance transform)")
    ax[1].axis("off")
    ax[2].imshow(out)
    ax[2].set_title(f"Підраховано об'єктів: {count}")
    ax[2].axis("off")
    plt.tight_layout()
    plt.savefig("figures/fig_count_objects.png", dpi=150)
    plt.close()
    return count


if __name__ == "__main__":
    df = file_parsing("Oschadbank (USD).xls")
    print("------------ Група 1: кластеризація числових даних ------------")
    print(df.describe().round(3))
    best_k, sil, Ks, labels = cluster_numeric(df)
    print("Силуетні коефіцієнти:", dict(zip(Ks, np.round(sil, 4))))
    print("Оптимальна кількість кластерів k =", best_k)
    uniq, cnt = np.unique(labels, return_counts=True)
    print("Розмір кластерів:", dict(zip(uniq.tolist(), cnt.tolist())))

    print("------------ Група 2: кластеризація зображення за кольором ------------")
    centers = cluster_image_color("images/color_scene.png", k=4)
    print("Центроїди кольорів (RGB):")
    print(centers)

    print("------------ Група 3: підрахунок об'єктів ------------")
    n_obj = count_objects("images/coins.png")
    print("Кількість виявлених об'єктів:", n_obj)
