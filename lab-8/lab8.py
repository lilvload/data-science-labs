import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)

SAMPLE_FILE = 'sample_data.xlsx'
DESCR_FILE = 'data_description.xlsx'
RANDOM_STATE = 42


def parse_description(file_name):
    d = pd.read_excel(file_name, sheet_name='Fields')
    d = d[['Field_in_data', 'Description_of_information', 'Filling']]
    return d


def parse_sample(file_name):
    d = pd.read_excel(file_name, sheet_name='Лист1')
    d.columns = [str(c).strip() for c in d.columns]
    return d


def select_indicators(df):
    mapping = {
        'loan_amount': 'loan_amount',
        'loan_days': 'loan_days',
        'gender_id': 'gender_id',
        'birth_date': 'birth_date',
        'Marital status': 'marital_status_id',
        'children_count_id': 'children_count_id',
        'education_id': 'education_id',
        'seniority_years': 'seniority_years',
        'monthly_income': 'monthly_income',
        'monthly_expenses': 'monthly_expenses',
        'other_loans_active': 'other_loans_active',
        'has_immovables': 'has_immovables',
        'has_movables': 'has_movables',
        'income_source_id': 'income_source_id',
        'employment_type_id': 'employment_type_id',
        'position_id': 'position_id',
        'organization_branch_id': 'organization_branch_id',
        'empoyees_count_id': 'empoyees_count_id',
        'fact_addr_owner_type_id': 'fact_addr_owner_type_id',
    }
    work = pd.DataFrame()
    for src, dst in mapping.items():
        if src in df.columns:
            work[dst] = df[src]
    work['loan_overdue'] = df['loan_overdue']
    work['overdue_days'] = df['overdue_days']
    return work


def clean_data(df):
    ref_date = pd.Timestamp('2023-01-01')
    df = df.copy()
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    df['age'] = (ref_date - df['birth_date']).dt.days / 365.25
    df.loc[(df['age'] < 18) | (df['age'] > 80), 'age'] = np.nan
    df = df.drop(columns=['birth_date'])

    numeric_cols = ['loan_amount', 'loan_days', 'seniority_years',
                    'monthly_income', 'monthly_expenses', 'other_loans_active', 'age']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

    cat_cols = ['gender_id', 'marital_status_id', 'children_count_id', 'education_id',
                'has_immovables', 'has_movables', 'income_source_id', 'employment_type_id',
                'position_id', 'organization_branch_id', 'empoyees_count_id',
                'fact_addr_owner_type_id']
    for c in cat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    df['monthly_income'] = df['monthly_income'].clip(lower=0)
    df['monthly_expenses'] = df['monthly_expenses'].clip(lower=0)
    return df


def engineer_features(df):
    df = df.copy()
    df['disposable_income'] = df['monthly_income'] - df['monthly_expenses']
    df['expense_ratio'] = df['monthly_expenses'] / df['monthly_income'].replace(0, np.nan)
    df['expense_ratio'] = df['expense_ratio'].fillna(1.0).clip(0, 3)
    df['payment_burden'] = (df['loan_amount'] / df['loan_days']) / \
                           df['monthly_income'].replace(0, np.nan)
    df['payment_burden'] = df['payment_burden'].fillna(df['payment_burden'].median())
    df['loan_to_income'] = df['loan_amount'] / df['monthly_income'].replace(0, np.nan)
    df['loan_to_income'] = df['loan_to_income'].fillna(df['loan_to_income'].median())
    return df


def build_target(df):
    return (df['loan_overdue'] == 0).astype(int)


def scoring_card(df):
    score = np.zeros(len(df))
    score += np.where(df['age'] >= 30, 20, np.where(df['age'] >= 24, 12, 5))
    score += np.where(df['monthly_income'] >= 15000, 25,
                      np.where(df['monthly_income'] >= 9000, 15, 6))
    score += np.where(df['expense_ratio'] <= 0.4, 20,
                      np.where(df['expense_ratio'] <= 0.7, 12, 4))
    score += np.where(df['seniority_years'] >= 8, 15,
                      np.where(df['seniority_years'] >= 3, 9, 3))
    score += np.where(df['other_loans_active'] == 0, 12,
                      np.where(df['other_loans_active'] <= 2, 6, 0))
    score += np.where(df['has_immovables'] == 1, 8, 0)
    score += np.where(df['education_id'] >= 4, 8,
                      np.where(df['education_id'] >= 3, 4, 0))
    score += np.where(df['loan_to_income'] <= 0.2, 12,
                      np.where(df['loan_to_income'] <= 0.4, 6, 0))
    return score


def detect_fraud(raw, clean):
    n = len(raw)
    flags = pd.DataFrame(index=raw.index)
    flags['anomaly_age'] = ((clean['age'] < 21) | (clean['age'] > 70))
    flags['income_too_high'] = clean['monthly_income'] > 100000
    flags['expenses_exceed_income'] = clean['monthly_expenses'] > clean['monthly_income']
    flags['zero_income'] = clean['monthly_income'] <= 0
    flags['seniority_vs_age'] = clean['seniority_years'] > (clean['age'] - 16)
    inc = clean['monthly_income']
    q1, q3 = inc.quantile(0.25), inc.quantile(0.75)
    iqr = q3 - q1
    flags['income_outlier'] = (inc > q3 + 3 * iqr) | (inc < q1 - 3 * iqr)
    flags['round_income'] = (clean['monthly_income'] % 10000 == 0) & \
                            (clean['monthly_income'] >= 30000)
    flags['heavy_burden'] = clean['payment_burden'] > 1.0
    flags['fraud_score'] = flags.sum(axis=1)
    flags['is_suspicious'] = flags['fraud_score'] >= 2
    return flags


def plot_score_distribution(scores, target):
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.hist(scores[target == 1], bins=25, alpha=0.6, label='Кредит повернено (1)', color='green')
    plt.hist(scores[target == 0], bins=25, alpha=0.6, label='Прострочення (0)', color='red')
    plt.xlabel('Інтегрована скорингова оцінка Scor')
    plt.ylabel('Кількість позичальників')
    plt.title('Розподіл скорингової оцінки за класами')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_score_dist.png', dpi=130)
    plt.close()


def plot_confusion(cm):
    plt.clf()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Прострочення (0)', 'Повернено (1)'])
    ax.set_yticklabels(['Прострочення (0)', 'Повернено (1)'])
    ax.set_xlabel('Прогноз моделі'); ax.set_ylabel('Фактичний клас')
    ax.set_title('Матриця невідповідностей LDA')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('fig_confusion.png', dpi=130)
    plt.close()


def plot_roc(y_true, y_proba, auc):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.clf()
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC-крива дискримінантної моделі')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('fig_roc.png', dpi=130)
    plt.close()


def plot_lda_projection(X_lda, y):
    lo, hi = np.percentile(X_lda, [1, 99])
    xc = np.clip(X_lda, lo, hi)
    bins = np.linspace(lo, hi, 35)
    plt.clf()
    plt.figure(figsize=(8, 4.5))
    plt.hist(xc[y == 1], bins=bins, alpha=0.6, label='Повернено (1)', color='green')
    plt.hist(xc[y == 0], bins=bins, alpha=0.6, label='Прострочення (0)', color='red')
    plt.xlabel('Значення дискримінантної функції LD1')
    plt.ylabel('Кількість позичальників')
    plt.title('Проєкція на дискримінантну вісь LDA')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_lda_proj.png', dpi=130)
    plt.close()


def plot_clusters(scores, decision):
    plt.clf()
    plt.figure(figsize=(8, 4.5))
    idx = np.arange(len(scores))
    order = np.argsort(scores)
    s_sorted = scores[order]
    d_sorted = decision[order]
    plt.scatter(idx[d_sorted == 1], s_sorted[d_sorted == 1], s=10,
                color='green', label='Надати кредит')
    plt.scatter(idx[d_sorted == 0], s_sorted[d_sorted == 0], s=10,
                color='red', label='Відмова')
    plt.xlabel('Позичальники (відсортовано за Scor)')
    plt.ylabel('Скорингова оцінка Scor')
    plt.title('Кластеризація позичальників: надання / відмова у кредитуванні')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig_clusters.png', dpi=130)
    plt.close()


def plot_fraud(flags):
    cols = ['anomaly_age', 'income_too_high', 'expenses_exceed_income', 'zero_income',
            'seniority_vs_age', 'income_outlier', 'round_income', 'heavy_burden']
    counts = [int(flags[c].sum()) for c in cols]
    labels = ['Вік поза\nнормою', 'Завищений\nдохід', 'Витрати >\nдоходу', 'Нульовий\nдохід',
              'Стаж >\nвіку', 'Викид\nдоходу', 'Округлений\nдохід', 'Високе\nнаванта-\nження']
    plt.clf()
    plt.figure(figsize=(9, 5))
    plt.bar(labels, counts, color='crimson', alpha=0.7)
    plt.ylabel('Кількість спрацювань')
    plt.title('Індикатори виявлення шахрайства та фальсифікації даних')
    plt.xticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('fig_fraud.png', dpi=130)
    plt.close()


if __name__ == '__main__':
    descr = parse_description(DESCR_FILE)
    print('Опис полів (фрагмент):')
    print(descr.head(10).to_string())

    raw = parse_sample(SAMPLE_FILE)
    print('\nРозмір вибірки:', raw.shape)

    work = select_indicators(raw)
    clean = clean_data(work)
    clean = engineer_features(clean)
    y = build_target(clean)

    print('\nБаланс класів (target):')
    print(y.value_counts())

    scores = scoring_card(clean)
    threshold = np.percentile(scores, 30)
    decision = (scores >= threshold).astype(int)
    print('\nПоріг скорингу Scor:', round(threshold, 2))
    print('Надано кредитів:', int(decision.sum()), 'з', len(decision))

    feature_cols = ['loan_amount', 'loan_days', 'age', 'gender_id', 'marital_status_id',
                    'children_count_id', 'education_id', 'seniority_years', 'monthly_income',
                    'monthly_expenses', 'other_loans_active', 'has_immovables', 'has_movables',
                    'income_source_id', 'disposable_income', 'expense_ratio',
                    'payment_burden', 'loan_to_income']
    feature_cols = [c for c in feature_cols if c in clean.columns]
    X = clean[feature_cols].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    y_proba = lda.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print('\n===== Метрики дискримінантної моделі (LDA) =====')
    print(f'Accuracy : {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall   : {rec:.4f}')
    print(f'F1-score : {f1:.4f}')
    print(f'ROC-AUC  : {auc:.4f}')
    print('Матриця невідповідностей:')
    print(cm)

    X_lda_all = lda.transform(Xs).ravel()

    flags = detect_fraud(raw, clean)
    print('\n===== Виявлення шахрайства =====')
    print('Підозрілих заявок (fraud_score>=2):', int(flags['is_suspicious'].sum()))
    print(flags[['anomaly_age', 'income_too_high', 'expenses_exceed_income',
                 'zero_income', 'seniority_vs_age', 'income_outlier',
                 'round_income', 'heavy_burden']].sum().to_string())

    plot_score_distribution(scores, y)
    plot_confusion(cm)
    plot_roc(y_test, y_proba, auc)
    plot_lda_projection(X_lda_all, y)
    plot_clusters(scores, decision)
    plot_fraud(flags)

    out = raw[['Application']].copy()
    out['Scor'] = np.round(scores, 1)
    out['decision'] = decision
    out['credit_returned_pred'] = lda.predict(Xs)
    out['is_suspicious'] = flags['is_suspicious'].values
    out.to_excel('scoring_results.xlsx', index=False)
    print('\nРезультати збережено у scoring_results.xlsx')

    metrics_df = pd.DataFrame({
        'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'],
        'Значення': [round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4), round(auc, 4)]
    })
    metrics_df.to_csv('metrics.csv', index=False)
    print(metrics_df.to_string())
