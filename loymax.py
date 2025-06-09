# Standard libraries
from typing import Dict, List, Optional, Tuple
import warnings

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Machine learning
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier
)
from sklearn.model_selection import (
    cross_val_score, 
    train_test_split
)
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    accuracy_score
)
from sklearn.decomposition import PCA

# Specialized libraries
import loymax as lm
from pylift.eval import UpliftEval

# Suppress warnings
warnings.filterwarnings('ignore')

# Попытка импорта pylift с обработкой ошибок
try:
    from pylift.eval import UpliftEval
    from pylift import TransformedOutcome
    PYLIFT_AVAILABLE = True
except ImportError:
    PYLIFT_AVAILABLE = False
    print("⚠️  PyLift не найден. Используются альтернативные методы.")

# Проверяем доступность PyLift
try:
    from pylift import TransformedOutcome
    PYLIFT_AVAILABLE = True
except ImportError:
    PYLIFT_AVAILABLE = False
    print("⚠️ PyLift не установлен. Некоторые функции будут недоступны.")

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels не установлен. Используем альтернативные методы.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 1. НАХОДИМ ПЕРИОДЫ СКРЫТЫХ АКЦИЙ
def find_hidden_campaigns(purchases):
    """Ищем аномальные дни по начислениям на карты"""
    daily_stats = purchases.groupby(purchases['Opetation_datetime'].dt.date).agg({
        'toCard_stand': 'sum',
        'fromCard_stand': 'sum',
        'Person_BKEY': 'nunique',
        'Amount_Cheque': 'sum'
    }).reset_index()
    
    # Находим выбросы (топ 5% дней по начислениям)
    threshold = daily_stats['toCard_stand'].quantile(0.95)
    campaign_days = daily_stats[daily_stats['toCard_stand'] > threshold]['Opetation_datetime']
    
    return campaign_days.tolist(), daily_stats

# 2. АНАЛИЗ ИЗМЕНЕНИЙ В ПОКУПКАХ
def analyze_purchase_changes(purchases, goods, campaign_days):
    """Сравниваем поведение до/во время/после кампаний"""
    
    results = []
    
    for campaign_day in campaign_days[:3]:  # Берем первые 3 кампании
        
        # Определяем периоды
        before_start = campaign_day - timedelta(days=7)
        before_end = campaign_day - timedelta(days=1)
        during_start = campaign_day
        during_end = campaign_day + timedelta(days=3)
        after_start = during_end + timedelta(days=1)
        after_end = after_start + timedelta(days=7)
        
        # Фильтруем данные по периодам
        periods = {
            'before': (before_start, before_end),
            'during': (during_start, during_end), 
            'after': (after_start, after_end)
        }
        
        for period_name, (start, end) in periods.items():
            mask = (purchases['Opetation_datetime'].dt.date >= start) & \
                   (purchases['Opetation_datetime'].dt.date <= end)
            
            period_data = purchases[mask].merge(goods, on='Goods_BKEY', how='left')
            
            # Считаем метрики
            metrics = {
                'campaign_date': campaign_day,
                'period': period_name,
                'unique_customers': period_data['Person_BKEY'].nunique(),
                'avg_basket_size': period_data.groupby('Purchase_ID')['Qnt'].sum().mean(),
                'avg_cheque': period_data['Amount_Cheque'].mean(),
                'unique_products': period_data['Goods_BKEY'].nunique(),
                'unique_categories': period_data['cat_lev_02_BKEY'].nunique()
            }
            
            results.append(metrics)
    
    return pd.DataFrame(results)

# 3. ТОП ИЗМЕНЕНИЯ В КАТЕГОРИЯХ
def analyze_category_shifts(purchases, goods, campaign_days):
    """Какие категории больше всего меняются во время кампаний"""
    
    category_changes = []
    
    for campaign_day in campaign_days[:2]:
        # До кампании
        before_mask = (purchases['Opetation_datetime'].dt.date >= campaign_day - timedelta(days=7)) & \
                     (purchases['Opetation_datetime'].dt.date < campaign_day)
        
        # Во время кампании  
        during_mask = (purchases['Opetation_datetime'].dt.date >= campaign_day) & \
                     (purchases['Opetation_datetime'].dt.date <= campaign_day + timedelta(days=3))
        
        before_data = purchases[before_mask].merge(goods, on='Goods_BKEY')
        during_data = purchases[during_mask].merge(goods, on='Goods_BKEY')
        
        # Считаем долю каждой категории
        before_cat = before_data.groupby('cat_lev_02_BKEY')['Amount'].sum()
        during_cat = during_data.groupby('cat_lev_02_BKEY')['Amount'].sum()
        
        before_pct = before_cat / before_cat.sum() * 100
        during_pct = during_cat / during_cat.sum() * 100
        
        # Находим наибольшие изменения
        change = during_pct - before_pct
        top_changes = change.abs().nlargest(5)
        
        for cat, change_val in top_changes.items():
            category_changes.append({
                'campaign_date': campaign_day,
                'category': cat,
                'change_percent': change_val,
                'before_share': before_pct.get(cat, 0),
                'during_share': during_pct.get(cat, 0)
            })
    
    return pd.DataFrame(category_changes)

# 4. ВИЗУАЛИЗАЦИЯ
def create_visualizations(daily_stats, analysis_results, category_changes):
    """Создаем простые графики"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График 1: Динамика начислений (поиск кампаний)
    axes[0,0].plot(daily_stats['Opetation_datetime'], daily_stats['toCard_stand'])
    axes[0,0].set_title('Начисления на карты по дням (пики = скрытые акции)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # График 2: Изменение среднего чека по периодам
    period_order = ['before', 'during', 'after']
    avg_by_period = analysis_results.groupby('period')['avg_cheque'].mean().reindex(period_order)
    axes[0,1].bar(avg_by_period.index, avg_by_period.values)
    axes[0,1].set_title('Средний чек: до/во время/после кампаний')
    
    # График 3: Размер корзины
    basket_by_period = analysis_results.groupby('period')['avg_basket_size'].mean().reindex(period_order)
    axes[1,0].bar(basket_by_period.index, basket_by_period.values)
    axes[1,0].set_title('Размер корзины: до/во время/после')
    
    # График 4: Топ изменений категорий
    if not category_changes.empty:
        top_cats = category_changes.nlargest(5, 'change_percent')
        axes[1,1].barh(range(len(top_cats)), top_cats['change_percent'])
        axes[1,1].set_yticks(range(len(top_cats)))
        axes[1,1].set_yticklabels([f'Cat_{int(x)}' for x in top_cats['category']])
        axes[1,1].set_title('Топ изменений категорий (%)')
    
    plt.tight_layout()
    plt.show()

class ATENIVAnalyzer:
    """
    Анализатор для оценки среднего эффекта воздействия (ATE) 
    с помощью NIV (Net Information Value) на данных A/B-тестов
    """
    
    def __init__(self, data: pd.DataFrame, 
                 treatment_col: str = 'treatment',
                 outcome_col: str = 'conversion',
                 feature_cols: List[str] = None):
        """
        Инициализация анализатора
        
        Args:
            data: DataFrame с данными
            treatment_col: название столбца с индикатором воздействия
            outcome_col: название столбца с целевой переменной
            feature_cols: список названий столбцов с признаками
        """
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        
        # Автоматически определяем признаки если не указаны
        if feature_cols is None:
            self.feature_cols = [f'f{i}' for i in range(12)]  # f0-f11
        else:
            self.feature_cols = feature_cols
            
        # Дополнительные переменные для IV анализа
        self.network_vars = ['visit', 'exposure'] if 'visit' in data.columns and 'exposure' in data.columns else []
        
        print(f"Инициализация анализатора:")
        print(f"   Размер данных: {data.shape}")
        print(f"   Столбец воздействия: {treatment_col}")
        print(f"   Столбец исхода: {outcome_col}")
        print(f"   Количество признаков: {len(self.feature_cols)}")
        print(f"   Инструментальные переменные: {self.network_vars}")
    
    def calculate_niv(self, feature_col: str, n_bins: int = 10) -> Dict:
        """
        Расчет Net Information Value (NIV) для признака
        
        Args:
            feature_col: название столбца с признаком
            n_bins: количество бинов для разбиения
            
        Returns:
            Dict с результатами NIV анализа
        """
        try:
            # Создаем бины для непрерывных переменных
            feature_values = self.data[feature_col].copy()
            
            # Если переменная непрерывная, разбиваем на бины
            if len(feature_values.unique()) > n_bins:
                feature_binned = pd.cut(feature_values, bins=n_bins, duplicates='drop')
            else:
                feature_binned = feature_values
            
            # Создаем таблицу контингентности
            contingency_table = pd.crosstab(
                feature_binned, 
                [self.data[self.treatment_col], self.data[self.outcome_col]]
            )
            
            # Вычисляем NIV для каждой группы
            niv_results = {}
            total_treated = self.data[self.data[self.treatment_col] == 1].shape[0]
            total_control = self.data[self.data[self.treatment_col] == 0].shape[0]
            
            for bin_value in contingency_table.index:
                # Получаем частоты для данного бина
                treated_pos = contingency_table.loc[bin_value, (1, 1)] if (1, 1) in contingency_table.columns else 0
                treated_neg = contingency_table.loc[bin_value, (1, 0)] if (1, 0) in contingency_table.columns else 0
                control_pos = contingency_table.loc[bin_value, (0, 1)] if (0, 1) in contingency_table.columns else 0
                control_neg = contingency_table.loc[bin_value, (0, 0)] if (0, 0) in contingency_table.columns else 0
                
                # Избегаем деления на ноль
                if treated_pos + treated_neg == 0 or control_pos + control_neg == 0:
                    continue
                
                # Вычисляем rates
                treated_rate = treated_pos / (treated_pos + treated_neg) if (treated_pos + treated_neg) > 0 else 0
                control_rate = control_pos / (control_pos + control_neg) if (control_pos + control_neg) > 0 else 0
                
                # Веса групп
                treated_weight = (treated_pos + treated_neg) / total_treated if total_treated > 0 else 0
                control_weight = (control_pos + control_neg) / total_control if total_control > 0 else 0
                
                # NIV для данного бина
                if control_rate > 0 and treated_rate > 0:
                    niv_bin = (treated_rate - control_rate) * np.log(treated_rate / control_rate)
                else:
                    niv_bin = 0
                
                niv_results[str(bin_value)] = {
                    'treated_rate': treated_rate,
                    'control_rate': control_rate,
                    'treated_weight': treated_weight,
                    'control_weight': control_weight,
                    'niv': niv_bin,
                    'lift': treated_rate - control_rate
                }
            
            # Суммарный NIV
            total_niv = sum([result['niv'] * result['treated_weight'] for result in niv_results.values()])
            
            return {
                'feature': feature_col,
                'total_niv': total_niv,
                'bin_results': niv_results,
                'n_bins': len(niv_results)
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка при расчете NIV для {feature_col}: {e}")
            return {'feature': feature_col, 'total_niv': 0, 'bin_results': {}, 'n_bins': 0}
    
    def calculate_ate_simple(self) -> Dict:
        """
        Простой расчет ATE (разность средних)
        """
        treated_outcome = self.data[self.data[self.treatment_col] == 1][self.outcome_col].mean()
        control_outcome = self.data[self.data[self.treatment_col] == 0][self.outcome_col].mean()
        
        ate = treated_outcome - control_outcome
        
        # Стандартная ошибка
        treated_var = self.data[self.data[self.treatment_col] == 1][self.outcome_col].var()
        control_var = self.data[self.data[self.treatment_col] == 0][self.outcome_col].var()
        n_treated = self.data[self.data[self.treatment_col] == 1].shape[0]
        n_control = self.data[self.data[self.treatment_col] == 0].shape[0]
        
        se = np.sqrt(treated_var / n_treated + control_var / n_control)
        
        # t-статистика
        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_treated + n_control - 2))
        
        return {
            'ate': ate,
            'se': se,
            't_stat': t_stat,
            'p_value': p_value,
            'ci_lower': ate - 1.96 * se,
            'ci_upper': ate + 1.96 * se,
            'treated_mean': treated_outcome,
            'control_mean': control_outcome
        }
    
    def two_stage_least_squares(self, instrument: str, sample_size: int = None) -> Dict:
        """
        Двухшаговый МНК с исправленным доступом к параметрам
        """
        if not STATSMODELS_AVAILABLE:
            return self._two_stage_sklearn(instrument, sample_size)
        
        try:
            data_sample = self.data.sample(n=sample_size) if sample_size else self.data
            
            # Первая стадия: treatment ~ instrument + controls
            X_first_stage = sm.add_constant(data_sample[self.feature_cols + [instrument]])
            first_stage = OLS(data_sample[self.treatment_col], X_first_stage).fit()
            
            # Получаем предсказанные значения treatment
            treatment_hat = first_stage.fittedvalues
            
            # Вторая стадия: outcome ~ treatment_hat + controls
            X_second_stage = sm.add_constant(pd.concat([treatment_hat, data_sample[self.feature_cols]], axis=1))
            second_stage = OLS(data_sample[self.outcome_col], X_second_stage).fit()
            
            # Исправленный доступ к параметрам
            param_names = second_stage.params.index
            treatment_param_name = param_names[1]  # Первый после константы
            
            ate_estimate = second_stage.params[treatment_param_name]
            ate_se = second_stage.bse[treatment_param_name]
            
            # F-статистика для проверки слабости инструмента
            f_stat = first_stage.fvalue
            
            return {
                'ate': ate_estimate,
                'se': ate_se,
                't_stat': ate_estimate / ate_se if ate_se > 0 else 0,
                'p_value': 2 * (1 - stats.t.cdf(abs(ate_estimate / ate_se), len(data_sample) - len(self.feature_cols) - 2)) if ate_se > 0 else 1,
                'first_stage_f': f_stat,
                'weak_instrument': f_stat < 10,
                'instrument': instrument
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка в 2SLS для {instrument}: {e}")
            return self._two_stage_sklearn(instrument, sample_size)
    
    def _two_stage_sklearn(self, instrument: str, sample_size: int = None) -> Dict:
        """
        Альтернативная реализация 2SLS через sklearn
        """
        try:
            data_sample = self.data.sample(n=sample_size) if sample_size else self.data
            
            # Первая стадия
            X_first = data_sample[self.feature_cols + [instrument]]
            y_first = data_sample[self.treatment_col]
            
            first_stage = LinearRegression()
            first_stage.fit(X_first, y_first)
            treatment_hat = first_stage.predict(X_first)
            
            # Вторая стадия
            X_second = np.column_stack([treatment_hat, data_sample[self.feature_cols]])
            y_second = data_sample[self.outcome_col]
            
            second_stage = LinearRegression()
            second_stage.fit(X_second, y_second)
            
            ate_estimate = second_stage.coef_[0]  # Коэффициент при treatment_hat
            
            # Простая оценка стандартной ошибки
            residuals = y_second - second_stage.predict(X_second)
            mse = np.mean(residuals**2)
            ate_se = np.sqrt(mse / len(data_sample))
            
            return {
                'ate': ate_estimate,
                'se': ate_se,
                't_stat': ate_estimate / ate_se if ate_se > 0 else 0,
                'p_value': 2 * (1 - stats.t.cdf(abs(ate_estimate / ate_se), len(data_sample) - len(self.feature_cols) - 2)) if ate_se > 0 else 1,
                'first_stage_f': None,
                'weak_instrument': False,
                'instrument': instrument
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка в sklearn 2SLS для {instrument}: {e}")
            return {'ate': 0, 'se': 1, 't_stat': 0, 'p_value': 1, 'instrument': instrument}
    
    def pylift_analysis(self, sample_size: int = None) -> Dict:
        """
        Анализ с помощью PyLift
        """
        if not PYLIFT_AVAILABLE:
            print("⚠️ PyLift недоступен")
            return {}
        
        try:
            data_sample = self.data.sample(n=sample_size) if sample_size else self.data
            
            # Подготовка данных для PyLift
            X = data_sample[self.feature_cols]
            y = data_sample[self.outcome_col]
            treatment = data_sample[self.treatment_col]
            
            # Создаем модель TransformedOutcome
            to_model = TransformedOutcome(X, y, treatment)
            to_model.fit()
            
            # Получаем предсказания
            predictions = to_model.predict(X)
            
            # Вычисляем ATE
            ate = predictions.mean()
            
            return {
                'ate': ate,
                'method': 'PyLift TransformedOutcome',
                'predictions': predictions,
                'model_score': to_model.score(X, y, treatment) if hasattr(to_model, 'score') else None
            }
            
        except Exception as e:
            print(f"⚠️ Ошибка в PyLift анализе: {e}")
            return {}
    # ..
    def comprehensive_niv_ate_analysis(self, sample_size: int = None) -> Dict:
        """
        Комплексный анализ ATE с использованием NIV
        """
        print("🔬 Начинаем комплексный NIV-ATE анализ...")
        
        results = {}
        
        # 1. Простой ATE
        print("\n1️. Простой расчет ATE...")
        simple_ate = self.calculate_ate_simple()
        results['simple_ate'] = simple_ate
        print(f"   ATE: {simple_ate['ate']:.6f} (SE: {simple_ate['se']:.6f}, p-value: {simple_ate['p_value']:.4f})")
        
        # 2. NIV для всех признаков
        print("\n2️. Расчет NIV для всех признаков...")
        niv_results = {}
        for feature in self.feature_cols:
    #         if feature in self.data.columns:
                niv_result = self.calculate_niv(feature)
                niv_results[feature] = niv_result
                print(f"   {feature}: NIV = {niv_result['total_niv']:.6f}")
        
        results['niv_analysis'] = niv_results
        
        # 3. Ранжирование признаков по NIV
        niv_ranking = sorted(niv_results.items(), key=lambda x: abs(x[1]['total_niv']), reverse=True)
        results['niv_ranking'] = niv_ranking
        
        print("\nТоп-5 признаков по NIV:")
        for i, (feature, niv_data) in enumerate(niv_ranking[:5]):
            print(f"   {i+1}. {feature}: {niv_data['total_niv']:.6f}")
        
        # 4. Инструментальные переменные (если доступны)
        if self.network_vars:
            print("\n3️. Анализ с инструментальными переменными...")
            iv_results = {}
            for instrument in self.network_vars:
                if instrument in self.data.columns:
                    iv_result = self.two_stage_least_squares(instrument, sample_size)
                    iv_results[instrument] = iv_result
                    print(f"   {instrument}: ATE = {iv_result['ate']:.6f} (SE: {iv_result['se']:.6f})")
            
            results['iv_analysis'] = iv_results
        
        # 5. PyLift анализ (если доступен)
        if PYLIFT_AVAILABLE:
            print("\n4️. PyLift анализ...")
            pylift_result = self.pylift_analysis(sample_size)
            if pylift_result:
                results['pylift_analysis'] = pylift_result
                print(f"   PyLift ATE: {pylift_result['ate']:.6f}")
        
        # 6. Сводка результатов
        print("\n📋 Сводка результатов:")
        print(f"   Простой ATE: {simple_ate['ate']:.6f}")
        
        if 'iv_analysis' in results:
            for instrument, iv_result in results['iv_analysis'].items():
                print(f"   2SLS ({instrument}): {iv_result['ate']:.6f}")
        
        if 'pylift_analysis' in results and results['pylift_analysis']:
            print(f"   PyLift ATE: {results['pylift_analysis']['ate']:.6f}")
        
        return results

class PyLiftNetworkAnalyzer:
    """
    Класс для анализа Network-based Weighted Outcome Estimation (NWOE) 
    с использованием PyLift
    """
    
    def __init__(self, data: pd.DataFrame, treatment_col: str = 'treatment', 
                 outcome_col: str = 'conversion', network_vars: List[str] = ['visit', 'exposure']):
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.network_vars = network_vars
        self.feature_cols = [col for col in data.columns 
                           if col.startswith('f') and col not in [treatment_col, outcome_col] + network_vars]
        
        # Проверка наличия необходимых колонок
        required_cols = [treatment_col, outcome_col] + network_vars
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")
        
        print(f"Инициализация PyLiftNetworkAnalyzer:")
        print(f"   Размер данных: {self.data.shape}")
        print(f"   Фичи: {len(self.feature_cols)}")
        print(f"   Сетевые переменные: {self.network_vars}")
    
    def calculate_network_weights(self, method: str = 'exposure_based') -> np.ndarray:
        """Вычисление весов на основе сетевых переменных"""
        
        if method == 'exposure_based':
            # Веса на основе exposure
            weights = 1 + self.data['exposure'].values
            
        elif method == 'visit_based':
            # Веса на основе visit
            weights = 1 + self.data['visit'].values
            
        elif method == 'combined':
            # Комбинированные веса
            exposure_norm = self.data['exposure'] / (self.data['exposure'].max() + 1e-8)
            visit_norm = self.data['visit'] / (self.data['visit'].max() + 1e-8)
            weights = 1 + 0.5 * (exposure_norm + visit_norm)
            
        elif method == 'inverse_variance':
            # Обратные веса дисперсии
            network_score = self.data['exposure'] + self.data['visit']
            network_var = np.var(network_score)
            if network_var > 0:
                weights = 1 / (1 + network_var * np.abs(network_score))
            else:
                weights = np.ones(len(self.data))
            
        else:
            weights = np.ones(len(self.data))
            
        # Нормализация весов
        weights = weights / (weights.sum() + 1e-8) * len(weights)
        return weights
    # ..
    def pylift_transformed_outcome_analysis(self, sample_size: int = None) -> Dict:# origin
        """
        Анализ с использованием Transformed Outcome подхода из PyLift
        """
        if not PYLIFT_AVAILABLE:
            return self._alternative_transformed_outcome_analysis(sample_size)
        
        print("Запуск PyLift Transformed Outcome анализа...")
        
        # Подготовка данных
        if sample_size and len(self.data) > sample_size:
            sample_data = self.data.sample(n=sample_size, random_state=42)
            print(f"📊 Используется выборка: {len(sample_data):,} из {len(self.data):,}")
        else:
            sample_data = self.data
        
        # Подготовка фичей с сетевыми переменными
        X_features = sample_data[self.feature_cols + self.network_vars].copy()
        
        # Заполнение пропущенных значений
        X_features = X_features.fillna(X_features.median())
        
        try:
            # Создание Transformed Outcome модели
            to_model = TransformedOutcome(
                df=sample_data,
                col_treatment=self.treatment_col,
                col_outcome=self.outcome_col,
                col_features=list(X_features.columns)
            )
            
            # Обучение модели
            to_model.fit()
            
            # Получение предсказаний uplift
            uplift_predictions = to_model.predict(X_features)
            
            # Вычисление ATE как среднего uplift
            ate_estimate = np.mean(uplift_predictions)
            
            print(f"✅ PyLift Transformed Outcome ATE: {ate_estimate:.6f}")
            
            return {
                'ate_estimate': ate_estimate,
                'uplift_predictions': uplift_predictions,
                'model': to_model,
                'method': 'pylift_transformed_outcome',
                'sample_size': len(sample_data)
            }
            
        except Exception as e:
            return self._alternative_transformed_outcome_analysis(sample_size)
    
    def _alternative_transformed_outcome_analysis(self, sample_size: int = None) -> Dict:# альтерн
        """
        Альтернативная реализация Transformed Outcome без PyLift
        """
        
        # Подготовка данных
        if sample_size and len(self.data) > sample_size:
            sample_data = self.data.sample(n=sample_size, random_state=42)
        else:
            sample_data = self.data
        
        # Подготовка фичей
        X = sample_data[self.feature_cols + self.network_vars].fillna(0)
        treatment = sample_data[self.treatment_col]
        outcome = sample_data[self.outcome_col]
        
        # Создание transformed outcome
        # TO = Y * T / p(T=1|X) - Y * (1-T) / p(T=0|X)
        
        # Оценка пропенсити скоров
        prop_model = LogisticRegression(random_state=42, max_iter=1000)
        prop_model.fit(X, treatment)
        propensity_scores = prop_model.predict_proba(X)[:, 1]
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
        
        # Вычисление transformed outcome
        transformed_outcome = (
            outcome * treatment / propensity_scores - 
            outcome * (1 - treatment) / (1 - propensity_scores)
        )
        
        # Модель для предсказания transformed outcome
        to_model = RandomForestRegressor(n_estimators=100, random_state=42)
        to_model.fit(X, transformed_outcome)
        
        # Предсказания uplift
        uplift_predictions = to_model.predict(X)
        
        # ATE как среднее uplift
        ate_estimate = np.mean(uplift_predictions)
        
        print(f"Transformed Outcome ATE: {ate_estimate:.6f}")
        
        return {
            'ate_estimate': ate_estimate,
            'uplift_predictions': uplift_predictions,
            'transformed_outcome': transformed_outcome,
            'propensity_model': prop_model,
            'uplift_model': to_model,
            'method': 'alternative_transformed_outcome',
            'sample_size': len(sample_data)
        }
    
    def pylift_evaluation_metrics(self, uplift_predictions: np.ndarray) -> Dict:
        """
        Вычисление метрик качества с использованием PyLift
        """
        if not PYLIFT_AVAILABLE:
            return self._alternative_evaluation_metrics(uplift_predictions)
        
        try:
            # Подготовка данных для оценки
            eval_data = self.data.copy()
            eval_data['uplift_pred'] = uplift_predictions
            
            # Создание объекта для оценки
            evaluator = UpliftEval(
                treatment_col=self.treatment_col,
                outcome_col=self.outcome_col,
                prediction_col='uplift_pred'
            )
            
            # Вычисление метрик
            metrics = evaluator.evaluate(eval_data)
            
            return {
                'pylift_metrics': metrics,
                'method': 'pylift_evaluation'
            }
            
        except Exception as e:
            print(f"⚠️  Ошибка в PyLift оценке: {e}")
            return self._alternative_evaluation_metrics(uplift_predictions)
    
    def _alternative_evaluation_metrics(self, uplift_predictions: np.ndarray) -> Dict:
        """
        Альтернативные метрики оценки качества
        """
        treatment = self.data[self.treatment_col].values
        outcome = self.data[self.outcome_col].values
        
        # Разделение на декили по uplift
        n_deciles = 10
        deciles = pd.qcut(uplift_predictions, q=n_deciles, labels=False, duplicates='drop')
        
        # Вычисление метрик по декилям
        decile_metrics = []
        for i in range(n_deciles):
            mask = deciles == i
            if mask.sum() > 0:
                treated_mask = mask & (treatment == 1)
                control_mask = mask & (treatment == 0)
                
                if treated_mask.sum() > 0 and control_mask.sum() > 0:
                    treated_rate = outcome[treated_mask].mean()
                    control_rate = outcome[control_mask].mean()
                    lift = treated_rate - control_rate
                    
                    decile_metrics.append({
                        'decile': i,
                        'size': mask.sum(),
                        'treated_rate': treated_rate,
                        'control_rate': control_rate,
                        'lift': lift
                    })
        
        # Общие метрики
        overall_metrics = {
            'mean_uplift_prediction': np.mean(uplift_predictions),
            'std_uplift_prediction': np.std(uplift_predictions),
            'decile_metrics': decile_metrics
        }
        
        return {
            'alternative_metrics': overall_metrics,
            'method': 'alternative_evaluation'
        }
    
    def network_weighted_pylift_analysis(self, sample_size: int = None) -> Dict:
        """
        Комбинированный анализ с сетевыми весами и PyLift
        """
        print("🕸️  Запуск Network-Weighted PyLift анализа...")
        
        # Подготовка данных
        if sample_size and len(self.data) > sample_size:
            sample_data = self.data.sample(n=sample_size, random_state=42)
        else:
            sample_data = self.data
        
        results = {}
        
        # 1. Базовый PyLift анализ
        pylift_result = self.pylift_transformed_outcome_analysis(sample_size)
        results['base_pylift'] = pylift_result
        
        # 2. Анализ с различными сетевыми весами
        weight_methods = ['exposure_based', 'visit_based', 'combined']
        
        for method in weight_methods:
            print(f"   Анализ с весами: {method}")
            
            # Вычисление весов
            weights = self.calculate_network_weights(method)
            
            if sample_size and len(self.data) > sample_size:
                weights = weights[:len(sample_data)]
            
            # Взвешенная оценка ATE
            treatment = sample_data[self.treatment_col].values
            outcome = sample_data[self.outcome_col].values
            
            treated_outcomes = outcome[treatment == 1]
            control_outcomes = outcome[treatment == 0]
            treated_weights = weights[treatment == 1]
            control_weights = weights[treatment == 0]
            
            if len(treated_outcomes) > 0 and len(control_outcomes) > 0:
                weighted_ate = (
                    np.average(treated_outcomes, weights=treated_weights) - 
                    np.average(control_outcomes, weights=control_weights)
                )
                
                results[f'weighted_{method}'] = {
                    'ate_estimate': weighted_ate,
                    'weights': weights,
                    'method': f'network_weighted_{method}'
                }
        
        # 3. Комбинированная оценка
        all_ates = [result['ate_estimate'] for result in results.values() 
                   if 'ate_estimate' in result]
        
        if all_ates:
            combined_ate = np.mean(all_ates)
            ate_std = np.std(all_ates)
            
            results['combined'] = {
                'ate_estimate': combined_ate,
                'ate_std': ate_std,
                'individual_estimates': all_ates,
                'method': 'combined_network_pylift'
            }
            
            print(f"✅ Комбинированная оценка ATE: {combined_ate:.6f} ± {ate_std:.6f}")
        
        return results

class NetworkInstrumentalVariablesPyLift:
    """
    Класс для анализа Network Instrumental Variables (NIV) с PyLift
    """
    
    def __init__(self, data: pd.DataFrame, treatment_col: str = 'treatment', 
                 outcome_col: str = 'conversion', network_vars: List[str] = ['visit', 'exposure']):
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.network_vars = network_vars
        self.feature_cols = [col for col in data.columns 
                           if col.startswith('f') and col not in [treatment_col, outcome_col] + network_vars]
        
        print(f"Инициализация NIV анализа:")
        print(f"   Инструменты: {self.network_vars}")
        print(f"   Фичи: {len(self.feature_cols)}")
    
    def check_instrument_validity(self) -> Dict:
        """Проверка валидности инструментальных переменных"""
        results = {}
        
        print("🔍 Проверка валидности инструментов...")
        
        for iv in self.network_vars:
            print(f"   Анализ инструмента: {iv}")
            
            # 1. Релевантность: корреляция IV с treatment
            relevance = self.data[iv].corr(self.data[self.treatment_col])
            
            # 2. Эксклюзивность: частная корреляция IV с outcome при контроле treatment
            # Используем остатки от регрессии outcome на treatment
            X_treatment = add_constant(self.data[self.treatment_col])
            outcome_model = OLS(self.data[self.outcome_col], X_treatment).fit()
            outcome_residuals = outcome_model.resid
            
            exclusivity = self.data[iv].corr(outcome_residuals)
            
            # 3. F-статистика для проверки силы инструмента
            X_iv = add_constant(self.data[iv])
            first_stage = OLS(self.data[self.treatment_col], X_iv).fit()
            f_stat = first_stage.fvalue
            
            # 4. Дополнительные тесты
            iv_treatment_r2 = first_stage.rsquared
            
            results[iv] = {
                'relevance': relevance,
                'exclusivity': abs(exclusivity),
                'f_statistic': f_stat,
                'first_stage_r2': iv_treatment_r2,
                'weak_instrument': f_stat < 10,  # Правило Staiger-Stock
                'first_stage_model': first_stage
            }
            
            print(f"      Релевантность: {relevance:.4f}")
            print(f"      Эксклюзивность: {abs(exclusivity):.4f}")
            print(f"      F-статистика: {f_stat:.2f}")
            print(f"      Слабый инструмент: {'Да' if f_stat < 10 else 'Нет'}")
        
        return results
    
    def two_stage_least_squares(self, instrument: str, sample_size: int = None) -> Dict:
        """
        Двухэтапный метод наименьших квадратов (2SLS)
        """
        print(f"🎯 2SLS анализ с инструментом: {instrument}")
        
        # Подготовка данных
        if sample_size and len(self.data) > sample_size:
            sample_data = self.data.sample(n=sample_size, random_state=42)
        else:
            sample_data = self.data
        
        # Подготовка переменных
        X_controls = sample_data[self.feature_cols].fillna(0)
        instrument_var = sample_data[instrument]
        treatment = sample_data[self.treatment_col]
        outcome = sample_data[self.outcome_col]
        
        # Этап 1: Регрессия treatment на инструмент и контроли
        X_first_stage = add_constant(pd.concat([instrument_var, X_controls], axis=1))
        first_stage = OLS(treatment, X_first_stage).fit()
        treatment_fitted = first_stage.fittedvalues
        
        # Этап 2: Регрессия outcome на fitted treatment и контроли
        X_second_stage = add_constant(pd.concat([treatment_fitted, X_controls], axis=1))
        second_stage = OLS(outcome, X_second_stage).fit()
        
        # ATE - это коэффициент при treatment в второй стадии
        ate_estimate = second_stage.params[1]  # Первый параметр после константы
        ate_se = second_stage.bse[1]
        
        print(f"   2SLS ATE: {ate_estimate:.6f} (SE: {ate_se:.6f})")
        
        return {
            'ate_estimate': ate_estimate,
            'ate_se': ate_se,
            'first_stage_model': first_stage,
            'second_stage_model': second_stage,
            'first_stage_r2': first_stage.rsquared,
            'second_stage_r2': second_stage.rsquared,
            'instrument': instrument,
            'method': '2sls',
            'sample_size': len(sample_data)
        }
    
    def pylift_instrumental_analysis(self, sample_size: int = None) -> Dict:
        """
        Комбинированный анализ NIV с PyLift методами
        """
        print("🔬 PyLift Instrumental Variables анализ...")
        
        results = {}
        
        # 1. Проверка валидности инструментов
        iv_validity = self.check_instrument_validity()
        results['instrument_validity'] = iv_validity
        
        # 2. 2SLS для каждого инструмента
        for instrument in self.network_vars:
            if not iv_validity[instrument]['weak_instrument']:
                tsls_result = self.two_stage_least_squares(instrument, sample_size)
                results[f'2sls_{instrument}'] = tsls_result
            else:
                print(f"⚠️  Пропуск слабого инструмента: {instrument}")
        
        # 3. Альтернативный подход через PyLift
        if PYLIFT_AVAILABLE:
            try:
                # Используем инструменты как дополнительные фичи
                pylift_analyzer = PyLiftNetworkAnalyzer(
                    self.data, self.treatment_col, self.outcome_col, self.network_vars
                )
                
                pylift_result = pylift_analyzer.pylift_transformed_outcome_analysis(sample_size)
                results['pylift_instrumental'] = pylift_result
                
            except Exception as e:
                print(f"⚠️  Ошибка в PyLift instrumental анализе: {e}")
        
        return results

def comprehensive_pylift_ate_analysis(data: pd.DataFrame, sample_size: int = None) -> Dict:
    """
    Комплексный анализ ATE с использованием PyLift и сетевых эффектов
    """
    print("🚀 КОМПЛЕКСНЫЙ PYLIFT ATE АНАЛИЗ")
    print("=" * 50)
    
    results = {}
    
    # 1. Базовые оценки
    print("📊 Базовые оценки...")
    naive_ate = (data[data['treatment'] == 1]['conversion'].mean() - 
                data[data['treatment'] == 0]['conversion'].mean())
    results['naive_ate'] = naive_ate
    print(f"   Наивная оценка ATE: {naive_ate:.6f}")
    
    # 2. PyLift Network анализ
    print("\n🕸️  PyLift Network анализ...")
    network_analyzer = PyLiftNetworkAnalyzer(data)
    pylift_results = network_analyzer.network_weighted_pylift_analysis(sample_size)
    results['pylift_network'] = pylift_results
    
    # 3. NIV анализ с PyLift
    print("\n🎯 Network Instrumental Variables анализ...")
    niv_analyzer = NetworkInstrumentalVariablesPyLift(data)
    niv_results = niv_analyzer.pylift_instrumental_analysis(sample_size)
    results['niv_analysis'] = niv_results
    
    # 4. Сравнение методов
    print("\n📈 Сравнение всех методов:")
    all_estimates = {'Наивная оценка': naive_ate}
    
    # Добавляем PyLift оценки
    if 'combined' in pylift_results:
        all_estimates['PyLift Combined'] = pylift_results['combined']['ate_estimate']
    
    if 'base_pylift' in pylift_results:
        all_estimates['PyLift Base'] = pylift_results['base_pylift']['ate_estimate']
    
    # Добавляем 2SLS оценки
    for key, value in niv_results.items():
        if key.startswith('2sls_') and 'ate_estimate' in value:
            instrument = key.replace('2sls_', '')
            all_estimates[f'2SLS ({instrument})'] = value['ate_estimate']
    
    # Вывод результатов
    for method, estimate in all_estimates.items():
        print(f"   {method:20}: {estimate:8.6f}")
    
    # Статистики
    if len(all_estimates) > 1:
        estimates_values = list(all_estimates.values())
        mean_estimate = np.mean(estimates_values)
        std_estimate = np.std(estimates_values)
        
        print(f"\n📊 Статистики:")
        print(f"   Среднее: {mean_estimate:.6f}")
        print(f"   Std: {std_estimate:.6f}")
        print(f"   Диапазон: {np.max(estimates_values) - np.min(estimates_values):.6f}")
        
        results['summary'] = {
            'all_estimates': all_estimates,
            'mean_estimate': mean_estimate,
            'std_estimate': std_estimate,
            'range': np.max(estimates_values) - np.min(estimates_values)
        }
    
    # 5. Рекомендации
    print("\n💡 Рекомендации:")
    
    if 'pylift_network' in results and 'combined' in results['pylift_network']:
        recommended_ate = results['pylift_network']['combined']['ate_estimate']
        recommended_std = results['pylift_network']['combined']['ate_std']
        
        print(f"   Рекомендуемая оценка ATE: {recommended_ate:.6f} ± {recommended_std:.6f}")
        print(f"   Метод: Комбинированный PyLift с сетевыми весами")
        
        results['recommendation'] = {
            'ate_estimate': recommended_ate,
            'ate_std': recommended_std,
            'method': 'Combined PyLift Network-Weighted'
        }
    
    return results

def quick_pylift_analysis(data: pd.DataFrame, sample_size: int = 50000) -> Dict:
    """
    Быстрый PyLift анализ для больших датасетов
    """
    print("⚡ БЫСТРЫЙ PYLIFT АНАЛИЗ")
    print("=" * 30)
    
    # 1. Наивная оценка
    naive_ate = (data[data['treatment'] == 1]['conversion'].mean() - 
                data[data['treatment'] == 0]['conversion'].mean())
    print(f"📊 Наивная оценка: {naive_ate:.6f}")
    
    # 2. Быстрый PyLift анализ
    analyzer = PyLiftNetworkAnalyzer(data)
    
    # Transformed Outcome
    to_result = analyzer.pylift_transformed_outcome_analysis(sample_size)
    
    # Network-weighted анализ
    network_result = analyzer.network_weighted_pylift_analysis(sample_size)
    
    results = {
        'naive_ate': naive_ate,
        'transformed_outcome': to_result['ate_estimate'],
        'network_weighted': network_result.get('combined', {}).get('ate_estimate', 'N/A')
    }
    
    print(f"📊 Результаты:")
    for method, estimate in results.items():
        if isinstance(estimate, (int, float)):
            print(f"   {method}: {estimate:.6f}")
        else:
            print(f"   {method}: {estimate}")
    
    return {
        'quick_estimates': results,
        'detailed_results': {
            'transformed_outcome': to_result,
            'network_weighted': network_result
        }
    }

# Функции для диагностики и визуализации остаются теми же
def network_effects_diagnostics(data: pd.DataFrame) -> Dict:
    """Диагностика сетевых эффектов"""
    
    print("\n🔬 ДИАГНОСТИКА СЕТЕВЫХ ЭФФЕКТОВ")
    print("=" * 40)
    
    diagnostics = {}
    
    # Корреляционный анализ
    network_vars = ['visit', 'exposure']
    corr_matrix = data[['treatment', 'conversion'] + network_vars].corr()
    
    print("📊 Корреляционная матрица:")
    print(corr_matrix.round(4))
    
    diagnostics['correlation_matrix'] = corr_matrix
    
    return diagnostics


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
# ..
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

# ..1 + статистика
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
    # ..
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
