import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, f1_score
import torch
import copy

# ===================== #
# 1. 数据加载与预处理
# ===================== #
columns = [
    'ID', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
    'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
    'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]

file_path = r"E:\google\Dataset\breast cancer\breast cancer\wdbc.data"
df = pd.read_csv(file_path, header=None, names=columns)
df['Diagnosis'] = LabelEncoder().fit_transform(df['Diagnosis'])

X = df.drop(['ID', 'Diagnosis'], axis=1)
y = df['Diagnosis']

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results_list = []
fold = 1

# ===================== #
# 2. K折交叉验证 + 重复训练取平均
# ===================== #
for train_idx, test_idx in kf.split(X, y):
    print(f"\n===== Fold {fold} =====")
    X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 划分验证集（注意：先划分再过采样，避免泄露）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, stratify=y_train_full, random_state=SEED
    )

    # 标准化（仅在训练集拟合）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # === SMOTE + Tomek Links ===
    from imblearn.combine import SMOTETomek
    smote_tomek = SMOTETomek(random_state=SEED)
    X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    # ==============================================

    # 用于集成预测
    test_pred_probas = []
    val_histories = []

    for repeat in range(3):
        print(f"  >> Repeat {repeat+1}/3")
        tabnet = TabNetClassifier(
            n_d=64, n_a=64, n_steps=6, gamma=1.5,
            lambda_sparse=5e-4, mask_type='entmax',
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-3, weight_decay=1e-4),
            scheduler_params={"T_max":100, "eta_min":1e-4},
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            seed=SEED+repeat, verbose=0
        )

        tabnet.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy'],
            patience=15,                # 更激进
            max_epochs=200,
            batch_size=128,
            virtual_batch_size=32,
            num_workers=0,
            drop_last=False
        )

        # 保存当前模型在测试集的预测
        y_test_proba = tabnet.predict_proba(X_test)[:, 1]
        test_pred_probas.append(y_test_proba)

        # 保存训练曲线
        val_histories.append(copy.deepcopy(tabnet.history))

    # 平均化预测结果
    avg_test_proba = np.mean(test_pred_probas, axis=0)

    # 验证集阈值优化（用最后一次训练的阈值）
    y_val_proba = tabnet.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    best_thr = thresholds[np.argmax(tpr - fpr)]

    # 评估
    y_test_pred = (avg_test_proba >= best_thr).astype(int)
    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    print(f"TabNet: Thr={best_thr:.3f} Acc={acc:.4f} Sens={sens:.4f} Spec={spec:.4f}")

    results_list.append({
        "Fold": fold,
        "Acc": acc, 
        "Sens": sens,
        "Spec": spec,
        "F1": f1,
        "Best_thr": best_thr
    })



    fold += 1

# ===================== #
# 3. 汇总结果
# ===================== #
results_df = pd.DataFrame(results_list)
print("\n===== 汇总结果 =====")
print(results_df)
print("\n平均结果：")
print(results_df.mean(numeric_only=True))
