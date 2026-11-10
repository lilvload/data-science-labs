import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
np.random.seed(42)

FIG = "figs/"


def model_sin(n, amplitude, period):
    t = np.arange(n)
    return amplitude * np.sin(2.0 * np.pi * t / period)


def random_norm(mean, sigma, n):
    return np.random.normal(mean, sigma, n)


def stat_characteristics(s, text):
    m = float(np.mean(s))
    d = float(np.var(s))
    sko = float(np.sqrt(d))
    print("------------", text, "-------------")
    print("математичне сподівання =", m)
    print("дисперсія =", d)
    print("СКВ =", sko)
    print("-----------------------------------------------------")
    return m, d, sko


def add_anomalies(s, percent, q_av, sigma):
    n = len(s)
    s_av = s.copy()
    count = int(n * percent / 100.0)
    idx = np.random.randint(0, n, count)
    for k in idx:
        s_av[k] = s[k] + np.random.uniform(-1.0, 1.0) * q_av * 3.0 * sigma
    return s_av, np.unique(idx), count


def rolling_median(s, window):
    n = len(s)
    half = window // 2
    base = np.zeros(n)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        base[i] = np.median(s[a:b])
    return base


def clean_anomalies(s, window, n_sigma):
    base = rolling_median(s, window)
    resid = s - base
    centre = np.median(resid)
    mad = np.median(np.abs(resid - centre)) * 1.4826
    if mad < 1e-9:
        mad = np.std(resid) + 1e-9
    mask = np.abs(resid - centre) > n_sigma * mad
    cleaned = s.copy()
    cleaned[mask] = base[mask]
    return cleaned, mask, base


def make_windows(s, w):
    x = np.zeros((len(s) - w, w))
    y = np.zeros(len(s) - w)
    for i in range(len(s) - w):
        x[i] = s[i:i + w]
        y[i] = s[i + w]
    return x, y


def build_net(arch, max_iter=3000):
    return MLPRegressor(hidden_layer_sizes=arch, activation="tanh", solver="adam",
                        learning_rate_init=0.005, max_iter=max_iter,
                        early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=50, tol=1e-7, random_state=42)


def count_params(net):
    total = 0
    for w in net.coefs_:
        total += w.size
    for b in net.intercepts_:
        total += b.size
    return int(total)


def recursive_forecast(net, scaler, seed_window, steps, w):
    win = list(seed_window)
    out = np.zeros(steps)
    for i in range(steps):
        p = net.predict(np.array(win[-w:]).reshape(1, -1))[0]
        out[i] = p
        win.append(p)
    return scaler.inverse_transform(out.reshape(-1, 1)).ravel()


if __name__ == "__main__":
    n = 5000
    amplitude = 100.0
    period = 500.0
    sigma = 15.0
    percent = 40
    q_av = 3
    w = 100
    train_ratio = 0.8

    ideal = model_sin(n, amplitude, period)
    noise = random_norm(0.0, sigma, n)
    sample = ideal + noise
    sample_av, anomaly_idx, anomaly_count = add_anomalies(sample, percent, q_av, sigma)

    stat_characteristics(noise, "Нормальний шум (m=0, СКВ=15)")
    stat_characteristics(sample, "Вибірка sin(x) + нормальний шум")
    m_av, d_av, sko_av = stat_characteristics(sample_av, "Вибірка з аномальними вимірами (40%)")

    cleaned, mask, base = clean_anomalies(sample_av, 31, 3.0)
    detected = int(mask.sum())
    stat_characteristics(cleaned, "Очищена вибірка (sliding window n-sigma)")

    plt.figure(figsize=(10, 4.2))
    plt.plot(sample_av, color="#9bbbd6", linewidth=0.7, label="Вибірка + аномалії")
    plt.plot(ideal, color="#d1620f", linewidth=1.4, label="Ідеальний тренд sin(x)")
    plt.xlabel("Дискретний відлік")
    plt.ylabel("Значення")
    plt.title("Синтезований dataset: sin(x) + нормальний шум + 40% аномалій")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG + "fig_dataset.png")
    plt.close()

    plt.figure(figsize=(10, 4.2))
    plt.plot(sample_av, color="#cdd5dd", linewidth=0.6, label="До очищення")
    plt.plot(cleaned, color="#1f6fb2", linewidth=0.9, label="Після очищення")
    plt.plot(ideal, color="#d1620f", linewidth=1.2, label="Ідеальний тренд")
    plt.xlabel("Дискретний відлік")
    plt.ylabel("Значення")
    plt.title("Очищення аномальних вимірів ковзним вікном за правилом n-sigma")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG + "fig_cleaned.png")
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.hist(noise, bins=30, facecolor="#3a78b5", alpha=0.75, edgecolor="white")
    plt.xlabel("Значення похибки")
    plt.ylabel("Частота")
    plt.title("Гістограма нормального закону розподілу випадкової складової")
    plt.tight_layout()
    plt.savefig(FIG + "fig_hist.png")
    plt.close()

    split = int(n * train_ratio)
    train_series = cleaned[:split]
    scaler = StandardScaler().fit(train_series.reshape(-1, 1))
    scaled = scaler.transform(cleaned.reshape(-1, 1)).ravel()

    x_all, y_all = make_windows(scaled, w)
    m_split = split - w
    x_train, y_train = x_all[:m_split], y_all[:m_split]
    x_test, y_test = x_all[m_split:], y_all[m_split:]

    net = build_net((64, 32))
    t0 = time.time()
    net.fit(x_train, y_train)
    train_time = time.time() - t0

    pred_test = scaler.inverse_transform(net.predict(x_test).reshape(-1, 1)).ravel()
    true_test = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    r2 = r2_score(true_test, pred_test)
    rmse = float(np.sqrt(mean_squared_error(true_test, pred_test)))
    mae = float(mean_absolute_error(true_test, pred_test))

    print("------------ Основна мережа (64, 32) -------------")
    print("кількість параметрів =", count_params(net))
    print("епох навчання =", net.n_iter_)
    print("час навчання, с =", round(train_time, 3))
    print("R2 =", round(r2, 4))
    print("RMSE =", round(rmse, 4))
    print("MAE =", round(mae, 4))
    print("-----------------------------------------------------")

    test_index = np.arange(split, n)

    plt.figure(figsize=(10, 4.2))
    plt.plot(test_index, true_test, color="#d1620f", linewidth=1.3, label="Фактичні значення")
    plt.plot(test_index, pred_test, color="#1f6fb2", linewidth=1.1, label="Прогноз ШНМ")
    plt.xlabel("Дискретний відлік")
    plt.ylabel("Значення")
    plt.title("Прогнозування на одну позицію вперед (тестова вибірка)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG + "fig_onestep.png")
    plt.close()

    seed_window = scaled[split - w:split]
    horizon = n - split
    rec = recursive_forecast(net, scaler, seed_window, horizon, w)
    true_full = scaler.inverse_transform(scaled[split:].reshape(-1, 1)).ravel()

    plt.figure(figsize=(10, 4.2))
    plt.plot(np.arange(split), cleaned[:split], color="#7a7a7a", linewidth=0.7, label="Навчальна вибірка")
    plt.plot(test_index, true_full, color="#d1620f", linewidth=1.3, label="Фактичні значення")
    plt.plot(test_index, rec, color="#1f6fb2", linewidth=1.2, label="Рекурсивний прогноз")
    plt.axvline(split, color="#444444", linestyle="--", linewidth=1.0)
    plt.xlabel("Дискретний відлік")
    plt.ylabel("Значення")
    plt.title("Багатокроковий рекурсивний прогноз часового ряду")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG + "fig_recursive.png")
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.plot(net.loss_curve_, color="#1f6fb2", linewidth=1.3)
    plt.xlabel("Епоха навчання")
    plt.ylabel("Значення функції втрат")
    plt.title("Крива навчання основної нейронної мережі (64, 32)")
    plt.tight_layout()
    plt.savefig(FIG + "fig_loss.png")
    plt.close()

    architectures = [(4,), (8,), (16,), (32,), (64,), (32, 16), (64, 32), (64, 32, 16), (128, 64)]
    rows = []
    for arch in architectures:
        m = build_net(arch)
        t0 = time.time()
        m.fit(x_train, y_train)
        tt = time.time() - t0
        p = scaler.inverse_transform(m.predict(x_test).reshape(-1, 1)).ravel()
        rows.append({
            "Структура": "(" + ", ".join(str(v) for v in arch) + ")",
            "Нейронів": sum(arch),
            "Шарів": len(arch),
            "Параметрів": count_params(m),
            "Епох": int(m.n_iter_),
            "Час, с": round(tt, 3),
            "R2": round(r2_score(true_test, p), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(true_test, p))), 4),
            "MAE": round(float(mean_absolute_error(true_test, p)), 4),
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv("metrics_table.csv", index=False)

    labels = table["Структура"].tolist()
    xpos = np.arange(len(labels))

    plt.figure(figsize=(9, 4.4))
    bars = plt.bar(xpos, table["R2"], color="#3a78b5", alpha=0.85)
    plt.xticks(xpos, labels, rotation=30, ha="right", fontsize=8)
    plt.ylim(0.900, 0.914)
    plt.ylabel("Коефіцієнт детермінації R2 (масштабовано)")
    plt.title("Залежність точності прогнозування (R2) від структури мережі")
    for b, v in zip(bars, table["R2"]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.0003, f"{v:.4f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG + "fig_struct_r2.png")
    plt.close()

    width = 0.4
    plt.figure(figsize=(9, 4.4))
    b1 = plt.bar(xpos - width / 2, table["RMSE"], width=width, color="#d1620f", alpha=0.85, label="RMSE")
    b2 = plt.bar(xpos + width / 2, table["MAE"], width=width, color="#3a78b5", alpha=0.85, label="MAE")
    plt.xticks(xpos, labels, rotation=30, ha="right", fontsize=8)
    plt.ylim(14.0, 23.5)
    plt.ylabel("Похибка прогнозування (масштабовано)")
    plt.title("Залежність похибок RMSE та MAE від структури мережі")
    for b, v in zip(b1, table["RMSE"]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=6.5)
    for b, v in zip(b2, table["MAE"]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=6.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG + "fig_struct_err.png")
    plt.close()

    best = table.loc[table["R2"].idxmax()]
    print("------------ Найкраща структура -------------")
    print(best.to_string())
    print("-----------------------------------------------------")

    summary = {
        "n": n, "amplitude": amplitude, "period": period, "sigma": sigma,
        "percent": percent, "anomaly_count": anomaly_count, "detected": detected,
        "window": w, "train_ratio": train_ratio, "split": split,
        "noise_stats": {"m": float(np.mean(noise)), "d": float(np.var(noise)), "sko": float(np.std(noise))},
        "av_stats": {"m": m_av, "d": d_av, "sko": sko_av},
        "main_net": {"arch": "(64, 32)", "params": count_params(net), "epochs": int(net.n_iter_),
                     "time": round(train_time, 3), "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)},
        "best_arch": {"structure": best["Структура"], "r2": float(best["R2"]),
                      "rmse": float(best["RMSE"]), "mae": float(best["MAE"]),
                      "neurons": int(best["Нейронів"]), "time": float(best["Час, с"])},
    }
    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("DONE")
