from system_of_Boolean_function2 import BooleanFunctionSystem
from itertools import combinations, product
import time
import matplotlib.pyplot as plt

# эксперимент 4 (рисунок 4) - зависимость времени работы алгортимов 1 и 2 от изменения мощности множеств k(сравнени времени работы оптимизированного и базового алгоритма)
def run_optimization_comparison_experiment():
    num_vars = 15
    num_functions = 5
    max_terms = 8
    num_systems = 5

    ks = range(1, 14)
    
    results = {
        'time': {
            'basic_original': [[] for _ in ks],
            'basic_optimized': [[] for _ in ks]
        }
    }
    
    with open('optimization_comparison_log.txt', 'w', encoding='utf-8') as log_file:
        log_file.write("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ОПТИМИЗИРОВАННЫХ И ОРИГИНАЛЬНЫХ АЛГОРИТМОВ\n")
        log_file.write("=" * 70 + "\n\n")

        for sys_id in range(num_systems):
            system = BooleanFunctionSystem(num_functions, num_vars, max_terms)
            print(f"Обработка системы {sys_id+1}/{num_systems}...")
            
            for k_idx, k in enumerate(ks):
                print(f"  k={k}...")
                
                start = time.perf_counter()
                basic_probs_orig = system.calculate_linearization_probability(k)
                time_basic_orig = time.perf_counter() - start
                
                start = time.perf_counter()
                basic_probs_opt = system.calculate_linearization_probability_optimized(k)
                time_basic_opt = time.perf_counter() - start
   
                
                results['time']['basic_original'][k_idx].append(time_basic_orig)
                results['time']['basic_optimized'][k_idx].append(time_basic_opt)
                
                if basic_probs_orig != basic_probs_opt:
                    error_msg = f" Различие в результатах базового алгоритма для k={k}, система {sys_id+1}"
                    print(error_msg)
                    log_file.write(error_msg + "\n")

        avg_times = {
            'basic_original': [],
            'basic_optimized': []
        }
        
        for k_idx, k in enumerate(ks):
            avg_basic_orig = sum(results['time']['basic_original'][k_idx]) / num_systems
            avg_basic_opt = sum(results['time']['basic_optimized'][k_idx]) / num_systems
            
            avg_times['basic_original'].append(avg_basic_orig)
            avg_times['basic_optimized'].append(avg_basic_opt)
            
            time_msg = (f"Время для k={k}: "
                       f"basic_orig={avg_basic_orig:.4f}s, "
                       f"basic_opt={avg_basic_opt:.4f}s")
            print(time_msg)
            log_file.write(time_msg + "\n")

    plt.figure(figsize=(12, 8))
    
    plt.plot(ks, avg_times['basic_original'], 'bo-', linewidth=2.5, markersize=8, label='Базовый алгоритм')
    plt.plot(ks, avg_times['basic_optimized'], 'ro--', linewidth=2.5, markersize=8, label='Оптимизированный базовый алгоритм')
    
    plt.xlabel('Размер подмножества (k)', fontsize=14)
    plt.ylabel('Время выполнения (с)', fontsize=14)
    plt.title('Сравнение времени выполнения алгоритмов линеаризации', fontsize=16, fontweight='bold')
    plt.xticks(ks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.legend(fontsize=13, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('optimization_time_comparison.png', dpi=300)
    plt.show()
    
    return avg_times

# эксперимент 1 (рисунок 1) - зависимость времени работы алгортима полного перебора, алгоритма 1, алгортима 1 с уточнениями от мощности множеств k
def run_speed_experiment():
    num_vars = 12
    num_functions = 6
    max_terms = 10
    num_systems = 1
    ks = range(2, 11)
    
    results = {
        'time': {
            'basic': [[] for _ in ks],
            'refined': [[] for _ in ks],
            'bruteforce': [[] for _ in ks]
        }
    }

    for sys_id in range(num_systems):
        system = BooleanFunctionSystem(num_functions, num_vars, max_terms)
        print(f"Обработка системы {sys_id+1}/{num_systems}...")
        
        for k_idx, k in enumerate(ks):
            print(f"  k={k}...")
            
            start = time.perf_counter()
            system.brute_force_linearization_check(k)
            time_bruteforce = time.perf_counter() - start
            
            start = time.perf_counter()
            system.calculate_linearization_probability(k)
            time_basic = time.perf_counter() - start
            
            start = time.perf_counter()
            system.calculate_refined_probability(k, max_iter=3)
            time_refined = time.perf_counter() - start
            
            results['time']['basic'][k_idx].append(time_basic)
            results['time']['refined'][k_idx].append(time_refined)
            results['time']['bruteforce'][k_idx].append(time_bruteforce)

    avg_results = {
        'time': {
            'basic': [],
            'refined': [],
            'bruteforce': []
        }
    }
    
    for k_idx, k in enumerate(ks):
        avg_time_basic = sum(results['time']['basic'][k_idx]) / num_systems
        avg_time_refined = sum(results['time']['refined'][k_idx]) / num_systems
        avg_time_brute = sum(results['time']['bruteforce'][k_idx]) / num_systems
        
        avg_results['time']['basic'].append(avg_time_basic)
        avg_results['time']['refined'].append(avg_time_refined)
        avg_results['time']['bruteforce'].append(avg_time_brute)
        
        print(f"Результаты для k={k}:")
        print(f"  Время: brute={avg_time_brute:.2f}s, basic={avg_time_basic:.2f}s, refined={avg_time_refined:.2f}s")

    plt.figure(figsize=(12, 8))
    
    plt.plot(ks, avg_results['time']['bruteforce'], 'r^-', label='Полный перебор', linewidth=2.5, markersize=8)
    plt.plot(ks, avg_results['time']['basic'], 'bo-', label='Базовый алгоритм', linewidth=2.5, markersize=8)
    plt.plot(ks, avg_results['time']['refined'], 'gs-', label='Уточненный алгоритм', linewidth=2.5, markersize=8)
    
    plt.xlabel('Размер подмножества (k)', fontsize=14)
    plt.ylabel('Время выполнения (с)', fontsize=14)
    plt.title('Сравнение времени выполнения алгоритмов линеаризации', fontsize=16, fontweight='bold')
    plt.xticks(ks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.legend(fontsize=13, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('time_comparison.png', dpi=300)
    plt.show()
    
    with open('time_experiment_results.txt', 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: ВРЕМЯ ВЫПОЛНЕНИЯ\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Параметры: num_vars={num_vars}, num_functions={num_functions}, "
                f"max_terms={max_terms}, num_systems={num_systems}\n\n")
        
        f.write("СРЕДНЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ (в секундах):\n")
        f.write("k\tbruteforce\tbasic\trefined\n")
        for k_idx, k in enumerate(ks):
            f.write(f"{k}\t{avg_results['time']['bruteforce'][k_idx]:.4f}\t"
                   f"{avg_results['time']['basic'][k_idx]:.4f}\t"
                   f"{avg_results['time']['refined'][k_idx]:.4f}\n")
    
    print(f"\nРезультаты сохранены в файл: time_experiment_results.txt")
    
    return avg_results

# эксперимент 3 (рисунок 3) - определение характера зависимости точности оценки максимальной вероятности линеаризации от мощности LP-множеств
def run_accuracy_experiment():
    num_vars = 12
    num_functions = 5
    max_terms = 10
    num_systems = 1
    ks = range(2, 11)
    
    results = {
        'max_prob_diff': {
            'basic': [],
            'refined': []
        }
    }

    for sys_id in range(num_systems):
        system = BooleanFunctionSystem(num_functions, num_vars, max_terms)
        print(f"Обработка системы {sys_id+1}/{num_systems}...")
        
        for k_idx, k in enumerate(ks):
            print(f"  k={k}...")
            
            brute_results, _ = system.brute_force_linearization_check(k)
            basic_results, _ = system.calculate_linearization_probability(k)
            refined_results, _ = system.calculate_refined_probability(k, max_iter=3)
            
            brute_max_prob = max(brute_results.values())
            basic_max_prob = max(basic_results.values())
            refined_max_prob = max(refined_results.values())
            
            max_prob_diff_basic = brute_max_prob - basic_max_prob
            max_prob_diff_refined = brute_max_prob - refined_max_prob
            
            if k_idx >= len(results['max_prob_diff']['basic']):
                results['max_prob_diff']['basic'].append([max_prob_diff_basic])
                results['max_prob_diff']['refined'].append([max_prob_diff_refined])
            else:
                results['max_prob_diff']['basic'][k_idx].append(max_prob_diff_basic)
                results['max_prob_diff']['refined'][k_idx].append(max_prob_diff_refined)
            
            print(f"    Макс.вероятность: brute={brute_max_prob:.4f}, "
                  f"basic={basic_max_prob:.4f}, "
                  f"refined={refined_max_prob:.4f}")
            print(f"    Разница: basic={max_prob_diff_basic:.6f}, "
                  f"refined={max_prob_diff_refined:.6f}")

    avg_results = {
        'max_prob_diff': {
            'basic': [],
            'refined': []
        }
    }
    
    for k_idx, k in enumerate(ks):
        avg_basic_diff = sum(results['max_prob_diff']['basic'][k_idx]) / num_systems
        avg_refined_diff = sum(results['max_prob_diff']['refined'][k_idx]) / num_systems
        
        avg_results['max_prob_diff']['basic'].append(avg_basic_diff)
        avg_results['max_prob_diff']['refined'].append(avg_refined_diff)
        
        print(f"\nРезультаты для k={k}:")
        print(f"  Средняя разница макс.вероятности:")
        print(f"    basic: {avg_basic_diff:.6f}")
        print(f"    refined: {avg_refined_diff:.6f}")

    plt.figure(figsize=(12, 8))
    
    plt.plot(ks, avg_results['max_prob_diff']['basic'], 'bo-', 
             label='Базовый алгоритм', linewidth=2.5, markersize=8)
    plt.plot(ks, avg_results['max_prob_diff']['refined'], 'gs-', 
             label='Уточненный алгоритм', linewidth=2.5, markersize=8)
    
    plt.xlabel('Размер подмножества (k)', fontsize=14)
    plt.ylabel('Разность: точное - приближенное', fontsize=14)
    plt.title('Точность оценки максимальной вероятности линеаризации', 
              fontsize=16, fontweight='bold')
    plt.xticks(ks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    
    plt.legend(fontsize=13, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('accuracy_max_prob_diff.png', dpi=300)
    plt.show()
    
    with open('accuracy_experiment_results.txt', 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: ТОЧНОСТЬ АЛГОРИТМОВ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Параметры: num_vars={num_vars}, num_functions={num_functions}, "
                f"max_terms={max_terms}, num_systems={num_systems}\n\n")
        
        f.write("РАЗНИЦА МАКСИМАЛЬНЫХ ВЕРОЯТНОСТЕЙ (точное - приближенное):\n")
        f.write("k\tbasic_diff\trefined_diff\n")
        for k_idx, k in enumerate(ks):
            f.write(f"{k}\t{avg_results['max_prob_diff']['basic'][k_idx]:.6f}\t"
                   f"{avg_results['max_prob_diff']['refined'][k_idx]:.6f}\n")
    
    print(f"\nРезультаты сохранены в файл: accuracy_experiment_results.txt")
    
    return avg_results

# эксперимент 2 (рисунок 2) -  зависимость времени работы алгортима полного перебора, алгоритма 1 , алгортима 1 с уточнениями от числа переменных n
def run_scalability_experiment():
    fixed_k = 5 
    n_values = range(7, 15, 1)
    num_functions = 10
    max_terms = 5
    num_systems = 1 
    
    results = {
        'time': {
            'bruteforce': [],
            'basic': [],
            'refined': []
        }
    }

    for n in n_values:
        print(f"\nАнализ для n = {n} (k={fixed_k})...")
        time_brute = []
        time_basic = []
        time_refined = []
        
        for sys_id in range(num_systems):
            system = BooleanFunctionSystem(num_functions, n, max_terms)
            
            start = time.perf_counter()
            system.brute_force_linearization_check(fixed_k)
            time_brute.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            system.calculate_linearization_probability(fixed_k)
            time_basic.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            system.calculate_refined_probability(fixed_k, max_iter=2)
            time_refined.append(time.perf_counter() - start)
            
            print(f"Система {sys_id+1} завершена", end="\r")
        
        results['time']['bruteforce'].append(sum(time_brute) / len(time_brute))
        results['time']['basic'].append(sum(time_basic) / len(time_basic))
        results['time']['refined'].append(sum(time_refined) / len(time_refined))
        
        print(f"n={n}: brute={results['time']['bruteforce'][-1]:.2f}s, "
              f"basic={results['time']['basic'][-1]:.4f}s, "
              f"refined={results['time']['refined'][-1]:.4f}s")

    plt.figure(figsize=(14, 8))
    
    plt.plot(n_values, results['time']['bruteforce'], 'r^-', 
             label='Полный перебор', linewidth=2.5, markersize=8)
    plt.plot(n_values, results['time']['basic'], 'bo-', 
             label='Базовый алгоритм', linewidth=2.5, markersize=8)
    plt.plot(n_values, results['time']['refined'], 'gs-', 
             label='Уточненный алгоритм', linewidth=2.5, markersize=8)
    
    plt.xlabel('Количество переменных (n)', fontsize=14)  
    plt.ylabel('Время выполнения (с)', fontsize=14)
    plt.title(f'Зависимость времени выполнения от количества переменных (k={fixed_k})', 
              fontsize=16, fontweight='bold')
    plt.xticks(n_values, fontsize=12)  
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    

    plt.legend(fontsize=13, loc='upper left', framealpha=0.9)
    
    
    plt.tight_layout()
    plt.savefig(f'scalability_k{fixed_k}.png', dpi=300)
    plt.show()
    

    with open('scalability_results.txt', 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: МАСШТАБИРУЕМОСТЬ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Параметры: фиксированный k={fixed_k}, num_functions={num_functions}, "
                f"max_terms={max_terms}, num_systems={num_systems}\n\n")
        
        f.write("СРЕДНЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ (в секундах):\n")
        f.write("n\tbruteforce\tbasic\trefined\n")  
        for i, n in enumerate(n_values):
            f.write(f"{n}\t{results['time']['bruteforce'][i]:.4f}\t"
                   f"{results['time']['basic'][i]:.6f}\t"
                   f"{results['time']['refined'][i]:.6f}\n")
    
    print(f"\nРезультаты сохранены в файл: scalability_results.txt")
    
    return results

# эксперимент (рисунки 5 и 6) для определения зависимости мощности множетсва EX и величниы EX/ALL от мощности множеств k
def run_random_system_analysis():
    
    num_vars = 20
    num_functions = 5
    max_terms = 10
    max_monomial_degree = 2
    num_systems = 50
    ks = range(2, 19)
    
    time_results = {
        'basic': [[] for _ in ks],
        'basic_optimized': [[] for _ in ks]
    }
    
    excluded_sets_data = {k: [] for k in ks}
    total_sets_counts = {k: len(list(combinations(range(1, num_vars + 1), k))) for k in ks}
    
    with open('random_system_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("АНАЛИЗ СЛУЧАЙНОЙ СИСТЕМЫ: ВРЕМЯ И EXCLUDED_SETS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Параметры: num_vars={num_vars}, num_functions={num_functions}, max_terms={max_terms}\n")
        f.write(f"Макс. степень монома: {max_monomial_degree}\n")
        f.write(f"Количество систем: {num_systems}\n\n")

        for sys_id in range(num_systems):
            system = BooleanFunctionSystem(num_functions, num_vars, max_terms,
                                         max_monomial_degree=max_monomial_degree)
            system.pretty_print()
            print(f"\nОбработка системы {sys_id+1}/{num_systems}...")
            f.write(f"\nСИСТЕМА {sys_id+1}:\n")
            
            for k_idx, k in enumerate(ks):
                print(f"  k={k}...")
                
                start = time.perf_counter()
                system.calculate_linearization_probability(k) 
                time_basic = time.perf_counter() - start
                
                start = time.perf_counter()
                _, _, excluded_count = system.calculate_linearization_probability_optimized(k)
                time_optimized = time.perf_counter() - start
                
                time_results['basic'][k_idx].append(time_basic)
                time_results['basic_optimized'][k_idx].append(time_optimized)
                excluded_sets_data[k].append(excluded_count)
                
                valid_sets_count = total_sets_counts[k] - excluded_count
                f.write(f"  k={k}: basic={time_basic:.4f}s, optimized={time_optimized:.4f}s, "
                       f"excluded={excluded_count}/{total_sets_counts[k]}, valid={valid_sets_count}\n")

    avg_times = {
        'basic': [],
        'basic_optimized': []
    }
    
    excluded_final = {}
    
    for k_idx, k in enumerate(ks):
        avg_basic = sum(time_results['basic'][k_idx]) / num_systems
        avg_optimized = sum(time_results['basic_optimized'][k_idx]) / num_systems
        avg_times['basic'].append(avg_basic)
        avg_times['basic_optimized'].append(avg_optimized)
        
        avg_excluded = sum(excluded_sets_data[k]) / num_systems
        excluded_ratio = avg_excluded / total_sets_counts[k]
        excluded_final[k] = {
            'avg_excluded_count': avg_excluded,
            'avg_ratio': excluded_ratio,
            'total_sets': total_sets_counts[k]
        }

    with open('random_system_analysis.txt', 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("ИТОГОВЫЕ СРЕДНИЕ ЗНАЧЕНИЯ\n")
        f.write("=" * 70 + "\n")
        f.write("k\tВсего\tИсключено\tДоля\tВремя_баз\tВремя_опт\n")
        
        for k_idx, k in enumerate(ks):
            excl_data = excluded_final[k]
            
            f.write(f"{k}\t{excl_data['total_sets']}\t{excl_data['avg_excluded_count']:.1f}\t"
                   f"{excl_data['avg_ratio']:.4f}\t{avg_times['basic'][k_idx]:.4f}\t"
                   f"{avg_times['basic_optimized'][k_idx]:.4f}\n")

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(ks, avg_times['basic'], 'bo-', label='Базовый алгоритм', linewidth=2, markersize=6)
    plt.plot(ks, avg_times['basic_optimized'], 'b--', label='Оптимизированный базовый алгоритм', linewidth=2, markersize=6)
    plt.xlabel('Размер подмножества (k)')
    plt.ylabel('Время выполнения (с)')
    plt.title('Сравнение времени выполнения')
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 3)
    excluded_values = [excluded_final[k]['avg_excluded_count'] for k in ks]
    plt.plot(ks, excluded_values, 'ro-', label='Мощность excluded_sets', linewidth=2, markersize=6)
    plt.xlabel('Размер подмножества (k)')
    plt.ylabel('Количество множеств')
    plt.title('Средняя мощность excluded_sets')
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    ratio_values = [excluded_final[k]['avg_ratio'] for k in ks]
    plt.plot(ks, ratio_values, 'g^-', label='Отношение excluded/all', linewidth=2, markersize=6)
    plt.xlabel('Размер подмножества (k)')
    plt.ylabel('Отношение')
    plt.title('Среднее отношение excluded_sets / all_sets')
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('random_system_analysis.png', dpi=300)
    plt.show()
    
    print(f"\nАнализ завершен!")
    print(f"Результаты сохранены в:")
    print(f"  - random_system_analysis.txt")
    print(f"  - random_system_analysis.png")
    
    return {
        'time_results': avg_times,
        'excluded_results': excluded_final
    }

if __name__ == "__main__":
   run_accuracy_experiment()
