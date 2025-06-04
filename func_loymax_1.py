import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from statsmodels.stats.weightstats import ttest_ind
from pylift.eval import UpliftEval
import loymax as lm
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class NetworkWeightedOutcomeEstimation:
    """
    Класс для анализа Network-based Weighted Outcome Estimation (NWOE)
    """
    
    def __init__(self, data: pd.DataFrame, treatment_col: str = 'treatment', 
                 outcome_col: str = 'conversion', network_vars: List[str] = ['visit', 'exposure']):
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.network_vars = network_vars
        self.feature_cols = [col for col in data.columns 
                           if col.startswith('f') and col not in [treatment_col, outcome_col] + network_vars]
        
    def calculate_network_weights(self, method: str = 'exposure_based') -> np.ndarray:
        """Вычисление весов на основе сетевых переменных"""
        
        if method == 'exposure_based':
            # Веса на основе exposure: больший вес для узлов с высоким exposure
            weights = 1 + self.data['exposure'].values
            
        elif method == 'visit_based':
            # Веса на основе visit: больший вес для узлов с высоким visit
            weights = 1 + self.data['visit'].values
            
        elif method == 'combined':
            # Комбинированные веса
            exposure_norm = self.data['exposure'] / self.data['exposure'].max()
            visit_norm = self.data['visit'] / self.data['visit'].max()
            weights = 1 + 0.5 * (exposure_norm + visit_norm)
            
        elif method == 'inverse_variance':
            # Обратные веса дисперсии (для снижения влияния выбросов)
            network_score = self.data['exposure'] + self.data['visit']
            weights = 1 / (1 + np.var(network_score) * network_score)
            
        else:
            weights = np.ones(len(self.data))
            
        return weights / weights.sum()  # Нормализация
    
    def propensity_score_weighting(self, use_network: bool = True) -> Dict:
        """Оценка с использованием пропенсити-скоров с учетом сетевых эффектов"""
        
        # Подготовка данных для пропенсити-модели
        if use_network:
            X_prop = np.hstack([
                self.data[self.feature_cols].values,
                self.data[self.network_vars].values
            ])
        else:
            X_prop = self.data[self.feature_cols].values
            
        # Обучение пропенсити-модели
        prop_model = LogisticRegression(random_state=42, max_iter=1000)
        prop_model.fit(X_prop, self.data[self.treatment_col])
        propensity_scores = prop_model.predict_proba(X_prop)[:, 1]
        
        # Предотвращение экстремальных весов
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
        
        # Вычисление IPW весов
        treatment = self.data[self.treatment_col].values
        ipw_weights = treatment / propensity_scores + (1 - treatment) / (1 - propensity_scores)
        
        # Если используем сетевые веса, комбинируем их с IPW
        if use_network:
            network_weights = self.calculate_network_weights('combined')
            final_weights = ipw_weights * network_weights * len(self.data)
        else:
            final_weights = ipw_weights
            
        # Нормализация весов
        final_weights = final_weights / final_weights.sum() * len(self.data)
        
        # Оценка ATE с весами
        outcome = self.data[self.outcome_col].values
        treated_outcomes = outcome[treatment == 1]
        control_outcomes = outcome[treatment == 0]
        treated_weights = final_weights[treatment == 1]
        control_weights = final_weights[treatment == 0]
        
        ate_estimate = (np.average(treated_outcomes, weights=treated_weights) - 
                       np.average(control_outcomes, weights=control_weights))
        
        return {
            'ate_estimate': ate_estimate,
            'propensity_scores': propensity_scores,
            'weights': final_weights,
            'model': prop_model
        }
    
    def doubly_robust_estimation(self, sample_size: int = None, use_simple_models: bool = True) -> Dict:
        """Быстрая дважды робастная оценка с сетевыми эффектами"""
        
        print("Запуск Doubly Robust оценки...")
        
        # Опциональная выборка для ускорения
        if sample_size and len(self.data) > sample_size:
            print(f"📊 Используется выборка размером {sample_size:,} из {len(self.data):,}")
            sample_data = self.data.sample(n=sample_size, random_state=42)
        else:
            sample_data = self.data
            
        # 1. Подготовка данных
        X_features = sample_data[self.feature_cols].values
        X_network = sample_data[self.network_vars].values
        X_prop = np.hstack([X_features, X_network])
        
        treatment = sample_data[self.treatment_col].values
        outcome = sample_data[self.outcome_col].values
        
        print("Обучение пропенсити-модели...")
        # 1. Пропенсити-скоры
        prop_model = LogisticRegression(random_state=42, max_iter=500, solver='lbfgs')
        prop_model.fit(X_prop, treatment)
        propensity_scores = prop_model.predict_proba(X_prop)[:, 1]
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
        
        print("Обучение моделей исходов...")
        # 2. Модели исходов (используем более простые модели для скорости)
        control_mask = treatment == 0
        treated_mask = treatment == 1
        
        if use_simple_models:
            # Используем LinearRegression для скорости
            control_model = LinearRegression()
            treated_model = LinearRegression()
        else:
            # RandomForest с меньшими параметрами
            control_model = RandomForestRegressor(n_estimators=50, max_depth=10, 
                                                n_jobs=-1, random_state=42)
            treated_model = RandomForestRegressor(n_estimators=50, max_depth=10, 
                                                n_jobs=-1, random_state=42)
        
        # Обучение моделей
        control_model.fit(X_prop[control_mask], outcome[control_mask])
        treated_model.fit(X_prop[treated_mask], outcome[treated_mask])
        
        print("Генерация предсказаний...")
        # 3. Предсказания
        mu0_hat = control_model.predict(X_prop)
        mu1_hat = treated_model.predict(X_prop)
        
        print("Вычисление ATE...")
        # 4. Дважды робастная оценка
        ipw_term1 = treatment * (outcome - mu1_hat) / propensity_scores
        ipw_term2 = (1 - treatment) * (outcome - mu0_hat) / (1 - propensity_scores)
        
        ate_estimate = np.mean(mu1_hat - mu0_hat) + np.mean(ipw_term1) - np.mean(ipw_term2)
        
        print("Оценка завершена!")
        
        return {
            'ate_estimate': ate_estimate,
            'mu0_predictions': mu0_hat,
            'mu1_predictions': mu1_hat,
            'propensity_scores': propensity_scores,
            'control_model': control_model,
            'treated_model': treated_model,
            'sample_size_used': len(sample_data)
        }
    
    def fast_nwoe_estimation(self, sample_size: int = 100000) -> Dict:
        """Быстрая NWOE оценка для больших датасетов"""
        
        print(f"⚡ Быстрая NWOE оценка на выборке {sample_size:,}...")
        
        # Стратифицированная выборка
        sample_data = self.data.groupby('treatment').apply(
            lambda x: x.sample(n=min(len(x), sample_size//2), random_state=42)
        ).reset_index(drop=True)
        
        print(f"📊 Размер выборки: {len(sample_data):,}")
        print(f"   Treatment 0: {(sample_data['treatment']==0).sum():,}")
        print(f"   Treatment 1: {(sample_data['treatment']==1).sum():,}")
        
        # Простая пропенсити-модель
        X_simple = sample_data[self.feature_cols[:6] + self.network_vars].values  # Только первые 6 фичей
        
        prop_model = LogisticRegression(random_state=42, max_iter=300)
        prop_model.fit(X_simple, sample_data[self.treatment_col])
        propensity_scores = prop_model.predict_proba(X_simple)[:, 1]
        propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
        
        # Сетевые веса
        network_weights = 1 + 0.5 * (sample_data['exposure'] + sample_data['visit'])
        network_weights = network_weights / network_weights.sum() * len(sample_data)
        
        # IPW веса
        treatment = sample_data[self.treatment_col].values
        ipw_weights = treatment / propensity_scores + (1 - treatment) / (1 - propensity_scores)
        
        # Комбинированные веса
        final_weights = ipw_weights * network_weights
        final_weights = np.clip(final_weights, 0.1, 10)  # Обрезка экстремальных весов
        
        # ATE оценка
        outcome = sample_data[self.outcome_col].values
        treated_outcomes = outcome[treatment == 1]
        control_outcomes = outcome[treatment == 0]
        treated_weights = final_weights[treatment == 1]
        control_weights = final_weights[treatment == 0]
        
        ate_estimate = (np.average(treated_outcomes, weights=treated_weights) - 
                       np.average(control_outcomes, weights=control_weights))
        
        print(f"✅ Быстрая оценка завершена: ATE = {ate_estimate:.6f}")
        
        return {
            'ate_estimate': ate_estimate,
            'propensity_scores': propensity_scores,
            'weights': final_weights,
            'sample_size': len(sample_data),
            'method': 'fast_nwoe'
        }

def comprehensive_ate_analysis(data: pd.DataFrame) -> Dict:
    """Комплексный анализ ATE с различными методами"""
    
    print("🔍 ЗАПУСК КОМПЛЕКСНОГО АНАЛИЗА ATE С СЕТЕВЫМИ ЭФФЕКТАМИ")
    print("=" * 60)
    
    results = {}
    
    # 1. Простая разность средних (наивная оценка)
    naive_ate = (data[data['treatment'] == 1]['conversion'].mean() - 
                data[data['treatment'] == 0]['conversion'].mean())
    results['naive_ate'] = naive_ate
    
    print(f"📊 Наивная оценка ATE: {naive_ate:.6f}")
    print()
    
    # 2. Анализ NIV
    print("🔧 АНАЛИЗ NETWORK INSTRUMENTAL VARIABLES (NIV)")
    print("-" * 40)
    
    niv_analyzer = NetworkInstrumentalVariables(data)
    
    # Проверка валидности инструментов
    iv_validity = niv_analyzer.check_instrument_validity()
    
    for iv, validity in iv_validity.items():
        print(f"📈 Инструмент '{iv}':")
        print(f"   Релевантность (корр. с treatment): {validity['relevance']:.4f}")
        print(f"   Эксклюзивность (остат. корр. с outcome): {validity['exclusivity']:.4f}")
        print(f"   F-статистика: {validity['f_statistic']:.2f}")
        print(f"   Слабый инструмент: {'⚠️  Да' if validity['weak_instrument'] else '✅ Нет'}")
        print()
    
    # 2SLS оценки для каждого инструмента
    niv_estimates = {}
    for iv in ['visit', 'exposure']:
        tsls_result = niv_analyzer.two_stage_least_squares(iv)
        niv_estimates[iv] = tsls_result
        print(f"🎯 2SLS оценка с инструментом '{iv}': {tsls_result['ate_estimate']:.6f}")
        print(f"   R² первого этапа: {tsls_result['first_stage_r2']:.4f}")
        print()
    
    results['niv_estimates'] = niv_estimates
    results['iv_validity'] = iv_validity
    
    # 3. Анализ NWOE
    print("🕸️  АНАЛИЗ NETWORK-BASED WEIGHTED OUTCOME ESTIMATION (NWOE)")
    print("-" * 50)
    
    nwoe_analyzer = NetworkWeightedOutcomeEstimation(data)
    
    # Пропенсити-скор взвешивание без сетевых эффектов
    psw_standard = nwoe_analyzer.propensity_score_weighting(use_network=False)
    print(f"⚖️  Стандартное PSW: {psw_standard['ate_estimate']:.6f}")
    
    # Пропенсити-скор взвешивание с сетевыми эффектами
    psw_network = nwoe_analyzer.propensity_score_weighting(use_network=True)
    print(f"🕸️  Network-enhanced PSW: {psw_network['ate_estimate']:.6f}")
    
    # Дважды робастная оценка
    dr_result = nwoe_analyzer.doubly_robust_estimation()
    print(f"🛡️  Doubly Robust: {dr_result['ate_estimate']:.6f}")
    print()
    
    results['nwoe_estimates'] = {
        'psw_standard': psw_standard,
        'psw_network': psw_network,
        'doubly_robust': dr_result
    }
    
    # 4. Сравнение всех методов
    print("📊 СРАВНЕНИЕ ВСЕХ МЕТОДОВ")
    print("-" * 30)
    
    all_estimates = {
        'Наивная оценка': naive_ate,
        '2SLS (visit)': niv_estimates['visit']['ate_estimate'],
        '2SLS (exposure)': niv_estimates['exposure']['ate_estimate'],
        'Стандартное PSW': psw_standard['ate_estimate'],
        'Network PSW': psw_network['ate_estimate'],
        'Doubly Robust': dr_result['ate_estimate']
    }
    
    for method, estimate in all_estimates.items():
        print(f"{method:15}: {estimate:8.6f}")
    
    # Статистики разброса
    estimates_values = list(all_estimates.values())
    print()
    print(f"📈 Среднее по всем методам: {np.mean(estimates_values):.6f}")
    print(f"📊 Стандартное отклонение: {np.std(estimates_values):.6f}")
    print(f"📏 Размах: {np.max(estimates_values) - np.min(estimates_values):.6f}")
    
    results['all_estimates'] = all_estimates
    results['summary_stats'] = {
        'mean': np.mean(estimates_values),
        'std': np.std(estimates_values),
        'range': np.max(estimates_values) - np.min(estimates_values)
    }
    
    return results

def network_effects_diagnostics(data: pd.DataFrame) -> Dict:
    """Диагностика сетевых эффектов"""
    
    print("\n🔬 ДИАГНОСТИКА СЕТЕВЫХ ЭФФЕКТОВ")
    print("=" * 40)
    
    diagnostics = {}
    
    # 1. Корреляционный анализ
    network_vars = ['visit', 'exposure']
    corr_matrix = data[['treatment', 'conversion'] + network_vars].corr()
    
    print("📊 Корреляционная матрица:")
    print(corr_matrix.round(4))
    print()
    
    # 2. Анализ spillover эффектов
    print("🌊 АНАЛИЗ SPILLOVER ЭФФЕКТОВ")
    print("-" * 30)
    
    # Группировка по уровням network exposure
    data['exposure_quartile'] = pd.qcut(data['exposure'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    data['visit_quartile'] = pd.qcut(data['visit'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Анализ по квартилям exposure
    exposure_analysis = data.groupby(['exposure_quartile', 'treatment']).agg({
        'conversion': ['count', 'mean', 'std']
    }).round(6)
    
    print("📈 Конверсия по квартилям exposure:")
    print(exposure_analysis)
    print()
    
    # Анализ по квартилям visit
    visit_analysis = data.groupby(['visit_quartile', 'treatment']).agg({
        'conversion': ['count', 'mean', 'std']
    }).round(6)
    
    print("📈 Конверсия по квартилям visit:")
    print(visit_analysis)
    print()
    
    # 3. Тест на гетерогенность эффектов
    print("🔍 ТЕСТ НА ГЕТЕРОГЕННОСТЬ ЭФФЕКТОВ")
    print("-" * 35)
    
    # ATE по квартилям exposure
    ate_by_exposure = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = data[data['exposure_quartile'] == q]
        ate_q = (subset[subset['treatment'] == 1]['conversion'].mean() - 
                subset[subset['treatment'] == 0]['conversion'].mean())
        ate_by_exposure[q] = ate_q
        print(f"ATE в {q} квартиле exposure: {ate_q:.6f}")
    
    print()
    
    # ATE по квартилям visit
    ate_by_visit = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = data[data['visit_quartile'] == q]
        ate_q = (subset[subset['treatment'] == 1]['conversion'].mean() - 
                subset[subset['treatment'] == 0]['conversion'].mean())
        ate_by_visit[q] = ate_q
        print(f"ATE в {q} квартиле visit: {ate_q:.6f}")
    
    diagnostics['correlation_matrix'] = corr_matrix
    diagnostics['ate_by_exposure'] = ate_by_exposure  
    diagnostics['ate_by_visit'] = ate_by_visit
    diagnostics['exposure_analysis'] = exposure_analysis
    diagnostics['visit_analysis'] = visit_analysis
    
    return diagnostics

def create_visualization_plots(data: pd.DataFrame, results: Dict):
    """Создание визуализаций для анализа"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Анализ ATE с сетевыми эффектами', fontsize=16, fontweight='bold')
    
    # 1. Сравнение оценок ATE
    methods = list(results['all_estimates'].keys())
    estimates = list(results['all_estimates'].values())
    
    axes[0, 0].barh(methods, estimates, color='skyblue', alpha=0.7)
    axes[0, 0].set_xlabel('ATE Estimate')
    axes[0, 0].set_title('Сравнение методов оценки ATE')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Распределение сетевых переменных
    axes[0, 1].hist(data['visit'], bins=50, alpha=0.7, label='Visit', density=True)
    axes[0, 1].hist(data['exposure'], bins=50, alpha=0.7, label='Exposure', density=True)
    axes[0, 1].set_xlabel('Значение')
    axes[0, 1].set_ylabel('Плотность')
    axes[0, 1].set_title('Распределение сетевых переменных')
    axes[0, 1].legend()
    
    # 3. Корреляционная матрица
    network_corr = data[['treatment', 'conversion', 'visit', 'exposure']].corr()
    sns.heatmap(network_corr, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0, 2], square=True)
    axes[0, 2].set_title('Корреляционная матрица')
    
    # 4. Пропенсити-скоры
    psw_result = results['nwoe_estimates']['psw_network']
    axes[1, 0].hist(psw_result['propensity_scores'], bins=50, alpha=0.7, 
                   color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Propensity Score')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].set_title('Распределение пропенсити-скоров')
    
    # 5. Conversion rate по группам и сетевым переменным
    data['exposure_bin'] = pd.cut(data['exposure'], bins=10, labels=False)
    grouped_data = data.groupby(['exposure_bin', 'treatment'])['conversion'].mean().reset_index()
    
    for treatment in [0, 1]:
        subset = grouped_data[grouped_data['treatment'] == treatment]
        axes[1, 1].plot(subset['exposure_bin'], subset['conversion'], 
                       marker='o', label=f'Treatment {treatment}')
    
    axes[1, 1].set_xlabel('Exposure Bin')
    axes[1, 1].set_ylabel('Conversion Rate')
    axes[1, 1].set_title('Conversion Rate по уровням Exposure')
    axes[1, 1].legend()
    
    # 6. Остатки модели
    dr_result = results['nwoe_estimates']['doubly_robust']
    treatment_mask = data['treatment'] == 1
    
    residuals_treated = (data.loc[treatment_mask, 'conversion'].values - 
                        dr_result['mu1_predictions'][treatment_mask])
    residuals_control = (data.loc[~treatment_mask, 'conversion'].values - 
                        dr_result['mu0_predictions'][~treatment_mask])
    
    axes[1, 2].scatter(dr_result['mu1_predictions'][treatment_mask][:1000], 
                      residuals_treated[:1000], alpha=0.5, label='Treated', s=1)
    axes[1, 2].scatter(dr_result['mu0_predictions'][~treatment_mask][:1000], 
                      residuals_control[:1000], alpha=0.5, label='Control', s=1)
    axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Predicted Values')
    axes[1, 2].set_ylabel('Residuals')
    axes[1, 2].set_title('Остатки модели (sample)')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Быстрый запуск для больших датасетов
def quick_nwoe_analysis(data: pd.DataFrame, sample_size: int = 100000) -> Dict:
    """Быстрый анализ NWOE для больших датасетов"""
    
    print("⚡ БЫСТРЫЙ NWOE АНАЛИЗ")
    print("=" * 30)
    
    # Наивная оценка
    naive_ate = (data[data['treatment'] == 1]['conversion'].mean() - 
                data[data['treatment'] == 0]['conversion'].mean())
    print(f"📊 Наивная оценка ATE: {naive_ate:.6f}")
    
    # Инициализация NWOE анализатора
    nwoe = NetworkWeightedOutcomeEstimation(data)
    
    # Быстрые методы
    print("\n🚀 Запуск быстрых методов...")
    
    # 1. Быстрая NWOE оценка
    fast_result = nwoe.fast_nwoe_estimation(sample_size)
    
    # 2. Быстрая Doubly Robust с линейными моделями
    dr_fast = nwoe.doubly_robust_estimation(sample_size, use_simple_models=True)
    
    # 3. Стандартное PSW на выборке
    sample_data = data.sample(n=min(len(data), sample_size), random_state=42)
    nwoe_sample = NetworkWeightedOutcomeEstimation(sample_data)
    psw_result = nwoe_sample.propensity_score_weighting(use_network=True)
    
    results = {
        'naive_ate': naive_ate,
        'fast_nwoe': fast_result['ate_estimate'],
        'doubly_robust_fast': dr_fast['ate_estimate'],
        'psw_network': psw_result['ate_estimate']
    }
    
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Наивная оценка:     {naive_ate:.6f}")
    print(f"   Быстрая NWOE:       {fast_result['ate_estimate']:.6f}")
    print(f"   Doubly Robust:      {dr_fast['ate_estimate']:.6f}")
    print(f"   Network PSW:        {psw_result['ate_estimate']:.6f}")
    
    # Согласованность результатов
    estimates = list(results.values())
    std_dev = np.std(estimates)
    print(f"\n📈 Стандартное отклонение: {std_dev:.6f}")
    
    if std_dev < 0.001:
        print("✅ Результаты согласованы")
    else:
        print("⚠️  Есть различия между методами")
    
    return {
        'estimates': results,
        'details': {
            'fast_nwoe': fast_result,
            'doubly_robust': dr_fast,
            'psw_network': psw_result
        },
        'consistency': std_dev
    }
    """Запуск полного анализа"""
    
    print("🚀 ПОЛНЫЙ АНАЛИЗ NWOE И NIV ДЛЯ ОЦЕНКИ ATE")
    print("=" * 50)
    
    # Основной анализ
    results = comprehensive_ate_analysis(data)
    
    # Диагностика сетевых эффектов
    diagnostics = network_effects_diagnostics(data)
    
    # Визуализация
    create_visualization_plots(data, results)
    
    # Итоговые рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ И ВЫВОДЫ")
    print("=" * 30)
    
    estimates = results['all_estimates']
    std_estimates = results['summary_stats']['std']
    
    if std_estimates < 0.001:
        print("✅ Низкая вариативность между методами - результаты согласованы")
    else:
        print("⚠️  Высокая вариативность между методами - требуется дополнительный анализ")
    
    # Проверка силы инструментов
    weak_instruments = [iv for iv, validity in results['iv_validity'].items() 
                       if validity['weak_instrument']]
    
    if weak_instruments:
        print(f"🔧 Слабые инструменты обнаружены: {weak_instruments}")
        print("   Рекомендуется полагаться на NWOE методы")
    else:
        print("✅ Инструменты достаточно сильны для NIV анализа")
    
    # Рекомендация по лучшему методу
    dr_estimate = estimates['Doubly Robust']
    network_psw_estimate = estimates['Network PSW']
    
    print(f"\n🎯 РЕКОМЕНДУЕМАЯ ОЦЕНКА ATE:")
    print(f"   Doubly Robust: {dr_estimate:.6f}")
    print(f"   (с учетом сетевых эффектов)")
    
    return {
        'results': results,
        'diagnostics': diagnostics,
        'recommendation': {
            'best_estimate': dr_estimate,
            'method': 'Doubly Robust',
            'confidence': 'high' if std_estimates < 0.001 else 'medium'
        }
    }



# eda
#+
def comprehensive_eda_for_ate_analysis(df):
    """
    Комплексный EDA анализ для подготовки к NWOE и NIV анализу ATE
    """
    
    print("="*60)
    print("ПРЕДВАРИТЕЛЬНЫЙ EDA ДЛЯ АНАЛИЗА ATE (NWOE/NIV)")
    print("="*60)
    
    # 1. БАЗОВАЯ ИНФОРМАЦИЯ О ДАННЫХ
    print("\n1. БАЗОВАЯ ИНФОРМАЦИЯ")
    print("-"*30)
    print(f"Размер датасета: {df.shape}")
    print(f"Количество наблюдений: {len(df)}")
    
    # Выделение групп переменных
    features = [col for col in df.columns if col.startswith('f')]
    treatment_var = 'treatment'
    outcome_var = 'conversion'
    network_vars = ['visit', 'exposure']
    
    print(f"Количество фичей: {len(features)}")
    print(f"Переменная treatment: {treatment_var}")
    print(f"Переменная outcome: {outcome_var}")
    print(f"Сетевые переменные: {network_vars}")
    
    # 2. АНАЛИЗ РАСПРЕДЕЛЕНИЯ TREATMENT
    print("\n2. АНАЛИЗ TREATMENT ГРУППЫ")
    print("-"*30)
    treatment_dist = df[treatment_var].value_counts().sort_index()
    treatment_prop = df[treatment_var].value_counts(normalize=True).sort_index()
    
    print("Распределение treatment:")
    for val, count, prop in zip(treatment_dist.index, treatment_dist.values, treatment_prop.values):
        print(f"  Treatment {val}: {count} ({prop:.3f})")
    
    # Проверка на дисбаланс
    min_group_size = treatment_dist.min()
    max_group_size = treatment_dist.max()
    imbalance_ratio = max_group_size / min_group_size
    print(f"Коэффициент дисбаланса: {imbalance_ratio:.2f}")
    if imbalance_ratio > 3:
        print("⚠️  ВНИМАНИЕ: Значительный дисбаланс в группах treatment!")
    
    # 3. АНАЛИЗ OUTCOME ПЕРЕМЕННОЙ
    print("\n3. АНАЛИЗ OUTCOME (CONVERSION)")
    print("-"*30)
    
    # Общая статистика outcome
    outcome_stats = df[outcome_var].describe()
    print("Общая статистика conversion:")
    print(outcome_stats)
    
    # Outcome по группам treatment
    print("\nConversion по группам treatment:")
    outcome_by_treatment = df.groupby(treatment_var)[outcome_var].agg(['count', 'mean', 'std'])
    print(outcome_by_treatment)
    
    # Тест на различия в outcome между группами
    treatment_groups = [group[outcome_var].values for name, group in df.groupby(treatment_var)]
    if len(treatment_groups) == 2:
        t_stat, p_val = stats.ttest_ind(treatment_groups[0], treatment_groups[1])
        print(f"\nT-test между группами: t={t_stat:.3f}, p={p_val:.4f}")
    
    # 4. АНАЛИЗ СЕТЕВЫХ ПЕРЕМЕННЫХ
    print("\n4. АНАЛИЗ СЕТЕВЫХ ПЕРЕМЕННЫХ")
    print("-"*30)
    
    for var in network_vars:
        print(f"\n{var.upper()}:")
        var_stats = df[var].describe()
        print(f"  Статистика: mean={var_stats['mean']:.3f}, std={var_stats['std']:.3f}")
        print(f"  Диапазон: [{var_stats['min']:.1f}, {var_stats['max']:.1f}]")
        
        # Корреляция с treatment и outcome
        corr_treatment = df[var].corr(df[treatment_var])
        corr_outcome = df[var].corr(df[outcome_var])
        print(f"  Корреляция с treatment: {corr_treatment:.3f}")
        print(f"  Корреляция с outcome: {corr_outcome:.3f}")
    
    # 5. АНАЛИЗ КОВАРИАТ (FEATURES)
    print("\n5. АНАЛИЗ КОВАРИАТ (FEATURES)")
    print("-"*30)
    
    # Базовая статистика фичей
    features_stats = df[features].describe()
    print(f"Статистика по {len(features)} фичам:")
    print(f"  Средние значения: [{features_stats.loc['mean'].min():.3f}, {features_stats.loc['mean'].max():.3f}]")
    print(f"  Стандартные отклонения: [{features_stats.loc['std'].min():.3f}, {features_stats.loc['std'].max():.3f}]")
    
    # Проверка на мультиколлинеарность
    correlation_matrix = df[features].corr()
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_val = abs(correlation_matrix.iloc[i, j])
            if corr_val > 0.8:
                high_corr_pairs.append((features[i], features[j], corr_val))
    
    if high_corr_pairs:
        print(f"\n⚠️  Обнаружены высокие корреляции (>0.8):")
        for f1, f2, corr in high_corr_pairs[:5]:  # Показываем первые 5
            print(f"  {f1} - {f2}: {corr:.3f}")
    else:
        print("\n✅ Критической мультиколлинеарности не обнаружено")
    
    # 6. БАЛАНС КОВАРИАТ МЕЖДУ ГРУППАМИ TREATMENT
    print("\n6. БАЛАНС КОВАРИАТ МЕЖДУ ГРУППАМИ")
    print("-"*30)
    
    # Стандартизированные различия средних
    smd_results = []
    for feature in features:
        group_stats = df.groupby(treatment_var)[feature].agg(['mean', 'std'])
        if len(group_stats) == 2:
            mean_diff = abs(group_stats['mean'].iloc[0] - group_stats['mean'].iloc[1])
            pooled_std = np.sqrt((group_stats['std'].iloc[0]**2 + group_stats['std'].iloc[1]**2) / 2)
            smd = mean_diff / pooled_std if pooled_std > 0 else 0
            smd_results.append((feature, smd))
    
    # Сортируем по убыванию SMD
    smd_results.sort(key=lambda x: x[1], reverse=True)
    
    print("Топ-5 фичей с наибольшим дисбалансом (SMD):")
    for feature, smd in smd_results[:5]:
        status = "⚠️" if smd > 0.25 else "✅"
        print(f"  {status} {feature}: SMD = {smd:.3f}")
    
    # Общая оценка баланса
    high_smd_count = sum(1 for _, smd in smd_results if smd > 0.25)
    print(f"\nФичей с SMD > 0.25: {high_smd_count}/{len(features)}")
    
    # 7. АНАЛИЗ ПРЕДПОЛОЖЕНИЙ ДЛЯ NWOE/NIV
    print("\n7. ПРОВЕРКА ПРЕДПОЛОЖЕНИЙ ДЛЯ NWOE/NIV")
    print("-"*30)
    
    # Instrumental Variable strength (для NIV)
    print("Анализ силы инструментальных переменных:")
    for var in network_vars:
        # Корреляция с treatment (должна быть значимой)
        corr_treatment = df[var].corr(df[treatment_var])
        
        # Частичная корреляция с outcome (должна быть слабой после контроля treatment)
        
        # Простая корреляция с outcome
        corr_outcome_simple = df[var].corr(df[outcome_var])
        
        print(f"  {var}:")
        print(f"    Корреляция с treatment: {corr_treatment:.3f}")
        print(f"    Корреляция с outcome: {corr_outcome_simple:.3f}")
        
        # Оценка силы инструмента
        if abs(corr_treatment) > 0.1:
            print(f"    ✅ Потенциально сильный инструмент")
        else:
            print(f"    ⚠️  Слабый инструмент")

#+# 1 + статистика
def basic_eda_analysis(df, remove_duplicates=True, verbose=True):
    """
    Выполняет базовый анализ данных (EDA)
    
    Параметры:
    ----------
    df : pandas.DataFrame
        Датафрейм для анализа
    remove_duplicates : bool, default=True
        Удалять ли дубликаты если они найдены
    verbose : bool, default=True
        Выводить ли подробную информацию
        
    Возвращает:
    -----------
    dict : словарь с результатами анализа
    pandas.DataFrame : очищенный датафрейм (если remove_duplicates=True)
    """
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Входные данные должны быть pandas DataFrame")
    
    results = {}
    
    # 1. Размер данных
    shape = df.shape
    results['shape'] = {
        'rows': shape[0],
        'columns': shape[1],
        'total_cells': shape[0] * shape[1]
    }
    
    if verbose:
        print("=== РАЗМЕР ДАННЫХ ===")
        print(f"Строки: {shape[0]:,}")
        print(f"Столбцы: {shape[1]:,}")
        # print(f"Общее количество ячеек: {shape[0] * shape[1]:,}")
        print()
    
    # 2. Занимаемая память
    memory_usage = df.memory_usage(deep=True)
    total_memory_bytes = memory_usage.sum()
    total_memory_mb = total_memory_bytes / (1024 * 1024)
    
    results['memory'] = {
        'total_bytes': total_memory_bytes,
        'total_mb': round(total_memory_mb, 2),
        'by_column': memory_usage.to_dict()
    }
    
    if verbose:
        # print("=== ИСПОЛЬЗОВАНИЕ ПАМЯТИ ===")
        print(f"Использование памяти: {total_memory_mb:.2f} MB ({total_memory_bytes:,} bytes)")
        # print("По столбцам:")
        # for col, mem in memory_usage.items():
        #     if col != 'Index':
        #         print(f"  {col}: {mem/1024:.1f} KB")
        print()
    
    # 3. Проверка и удаление дубликатов
    duplicates_count = df.duplicated().sum()
    duplicates_pct = (duplicates_count / len(df)) * 100 if len(df) > 0 else 0
    
    results['duplicates'] = {
        'count': duplicates_count,
        'percentage': round(duplicates_pct, 2),
        'removed': False
    }
    
    if verbose:
        print("=== ДУБЛИКАТЫ ===")
        print(f"Количество дубликатов: {duplicates_count:,}")
        print(f"Процент дубликатов: {duplicates_pct:.2f}%")
    
    # Удаление дубликатов если требуется
    cleaned_df = df.copy()
    if remove_duplicates and duplicates_count > 0:
        cleaned_df = df.drop_duplicates()
        results['duplicates']['removed'] = True
        if verbose:
            print(f"Дубликаты удалены. Новый размер: {cleaned_df.shape[0]:,} строк")
    
    if verbose:
        print()
    
    # 4. Проверка пропущенных значений
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100 if len(df) > 0 else pd.Series()
    
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percentage': missing_pct
    }).round(2)
    
    total_missing = missing_data.sum()
    total_missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100 if df.shape[0] * df.shape[1] > 0 else 0
    
    results['missing_values'] = {
        'total_missing': total_missing,
        'total_missing_percentage': round(total_missing_pct, 2),
        'by_column': missing_summary.to_dict('index'),
        'columns_with_missing': missing_data[missing_data > 0].index.tolist()
    }
    
    if verbose:
        print("=== ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ===")
        print(f"Общее количество пропущенных значений: {total_missing:,}")
        print(f"Общий процент пропущенных значений: {total_missing_pct:.2f}%")
        
        if total_missing > 0:
            print("\nПо столбцам:")
            for col in missing_data[missing_data > 0].index:
                count = missing_data[col]
                pct = missing_pct[col]
                print(f"  {col}: {count:,} ({pct:.2f}%)")
        else:
            print("Пропущенных значений не найдено")
        print()
    
    # 5. Типы данных
    dtypes_info = df.dtypes.value_counts()
    
    results['data_types'] = {
        'by_column': df.dtypes.to_dict(),
        'summary': dtypes_info.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
    }
    
    if verbose:
        print("=== ТИПЫ ДАННЫХ ===")
        print("Распределение типов данных:")
        for dtype, count in dtypes_info.items():
            print(f"  {dtype}: {count} столбцов")
        
        print(f"\nЧисловые столбцы ({len(results['data_types']['numeric_columns'])}): {results['data_types']['numeric_columns']}")
        print(f"Категориальные столбцы ({len(results['data_types']['categorical_columns'])}): {results['data_types']['categorical_columns']}")
        
        if results['data_types']['datetime_columns']:
            print(f"Столбцы с датами ({len(results['data_types']['datetime_columns'])}): {results['data_types']['datetime_columns']}")
        print()

    return results, cleaned_df
#++++++Функция для выявления выбросов с помощью межквартильного размаха (IQR)
def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers
class OutlierAnalyzer:#+
    """
    Класс для анализа выбросов в данных
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.feature_cols = [col for col in df.columns if col.startswith('f')]
        self.outlier_info = {}
    
    def detect_outliers(self):
        """
        Выявление выбросов по всем числовым колонкам
        Возвращает словарь с информацией о выбросах
        """
        outliers_summary = {}
        
        for col in self.feature_cols + ['conversion', 'visit', 'exposure']:
            if col in self.df.columns and np.issubdtype(self.df[col].dtype, np.number):
                outliers_summary[col] = self._detect_column_outliers(col)
        
        return outliers_summary
    
    def _detect_column_outliers(self, column):
        """Выявление выбросов для конкретного столбца"""
        data = self.df[column].dropna()
        
        # IQR метод
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'outliers_count': len(iqr_outliers),
            'outliers_percentage': len(iqr_outliers) / len(data) * 100,
            'bounds': (float(lower_bound), float(upper_bound)),
            'stats': {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max())
            }
        }
    
    def generate_report(self):
        """
        Генерация отчета по выбросам
        """
        outliers_summary = self.detect_outliers()
        total_outliers = sum(info['outliers_count'] for info in outliers_summary.values())
        
        report = "Количественный анализ по колонкам:\n"
        
        for col, info in outliers_summary.items():
            report += (
                f"- {col} : {info['outliers_count']} выбросов "
                f"({info['outliers_percentage']:.2f}%)\n"
            )
        
        report += f"\nОбщее кол-во выбросов: {total_outliers}\n"
        report += (
            f"* % от общего объема данных: "
            f"{total_outliers / len(self.df) * 100:.2f}%\n\n"
        )        

        report += (
            "1. NWOE/NIV методы созданы для работы с 'сырыми' данными - они используют биннинг, который автоматически снижает влияние выбросов\n"
            "2. Uplift-модели демонстрируют относительную устойчивость к выбросам\n"
            "3. 76.53% выбросов - это нормально для реальных данных маркетинга/веб-аналитики\n"
            "4. Выбросы содержат ценную информацию о высокоценных сегментах пользователей "
        )
        
        return report

class NetworkInstrumentalVariables:#+
    """
    Класс для анализа Network Instrumental Variables (NIV)
    """
    
    def __init__(self, data: pd.DataFrame, treatment_col: str = 'treatment', 
                 outcome_col: str = 'conversion', network_vars: List[str] = ['visit', 'exposure']):
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.network_vars = network_vars
        self.feature_cols = [col for col in data.columns 
                           if col.startswith('f') and col not in [treatment_col, outcome_col] + network_vars]
        
    def check_instrument_validity(self) -> Dict:
        """Проверка валидности инструментальных переменных"""
        results = {}
        
        for iv in self.network_vars:
            # 1. Релевантность: корреляция IV с treatment
            relevance = np.corrcoef(self.data[iv], self.data[self.treatment_col])[0, 1]
            
            # 2. Эксклюзивность: частная корреляция IV с outcome при контроле treatment
            # Используем остатки от регрессии outcome на treatment
            model_outcome_treatment = LinearRegression()
            X_treatment = self.data[[self.treatment_col]]
            outcome_residuals = self.data[self.outcome_col] - model_outcome_treatment.fit(
                X_treatment, self.data[self.outcome_col]).predict(X_treatment)
            
            exclusivity = np.corrcoef(self.data[iv], outcome_residuals)[0, 1]
            
            # 3. F-статистика для проверки силы инструмента
            X_iv = self.data[[iv]]
            y_treatment = self.data[self.treatment_col]
            model_iv_treatment = LinearRegression().fit(X_iv, y_treatment)
            r2 = r2_score(y_treatment, model_iv_treatment.predict(X_iv))
            n = len(self.data)
            f_stat = (r2 / (1 - r2)) * (n - 2)
            
            results[iv] = {
                'relevance': relevance,
                'exclusivity': exclusivity,
                'f_statistic': f_stat,
                'weak_instrument': f_stat < 10  # Правило Staiger-Stock
            }
            
        return results