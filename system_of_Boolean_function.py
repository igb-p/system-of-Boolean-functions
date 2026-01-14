import random
import copy
from itertools import combinations, product
 
def calculate_weight32(x):
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x3F


def calculate_weight(x):
    w = 0
    while x:
        w += calculate_weight32(x & 0xFFFFFFFF)
        x >>= 32
    return w


class BooleanFunctionSystem:
    def __init__(self, num_functions, num_vars, max_terms, functions=None, max_monomial_degree=None):
        if num_functions <= 0:
            raise ValueError("Количество функций должно быть больше 0.")
        if num_vars <= 0:
            raise ValueError("Количество переменных в системе должно быть больше 0.")
        if max_terms <= 0:
            raise ValueError("Максимальное количество мономов в функции должно быть больше 0.")

        self.num_functions = num_functions  # число функций
        self.num_vars = num_vars  # число переменных
        self.max_terms = max_terms  # максимальное число мономов в функции
        self.max_monomial_degree = max_monomial_degree # максимальная степень мономов системы
        
        if functions is not None:
            self.functions = functions
        else:
            self.functions = self._generate_system()

    def _generate_system(self):
        functions = []
        for _ in range(self.num_functions):
            function = self._generate_function()
            functions.append(function)
        return functions

    def _generate_function(self):  # генератор случайной функции
        #num_terms = random.randint(1, self.max_terms)
        num_terms = self.max_terms
        function = set()

        while len(function) < num_terms:
            monomial = self._generate_monomial()
            function.add(monomial)

        return list(function)

    def _generate_monomial(self):  # генератор случайного монома
        if self.max_monomial_degree is None:
            num_vars_in_monomial = random.randint(1, self.num_vars)
        else:
            if self.max_monomial_degree > self.num_vars or self.max_monomial_degree <= 0:
                raise ValueError(f"Невозможно создать моном степени {self.max_monomial_degree} из {self.num_vars} переменных")
            num_vars_in_monomial = random.randint(1, self.max_monomial_degree)

        available_vars = list(range(1, self.num_vars + 1))
        chosen_vars = random.sample(available_vars, num_vars_in_monomial)
    
        # Собираем моном
        monomial = 1
        for var in chosen_vars:
            monomial |= (1 << var)

        return monomial

    def copy(self):
        new_system = BooleanFunctionSystem(
            num_functions=self.num_functions,
            num_vars=self.num_vars,
            max_terms=self.max_terms
        )
        new_system.functions = copy.deepcopy(self.functions)
        return new_system

    def is_linear(self):  # проверка системы на линейность
        for function in self.functions:
            for monomial in function:
                if calculate_weight(monomial) > 2:
                    return False
        return True

    def pretty_print(self):  # вывод системы на экран
        functions_str = []
        for function in self.functions:
            monomials_str = []
            for monomial in function:
                if monomial == 0:
                    monomials_str.append("0")
                elif monomial == 1:
                    monomials_str.append("1")
                else:
                    terms = [f"x{i}" for i in range(1, self.num_vars + 1) if monomial & (1 << i)]
                    if terms:
                        monomials_str.append("".join(terms))
            
            functions_str.append(f"{' + '.join(monomials_str)}")
        print("\n".join(functions_str))

    @staticmethod
    def from_console():  # ввод системы функций (мономы представлены в виде чисел)
        num_functions = int(input("Введите количество функций: "))
        if num_functions <= 0:
            raise ValueError("Количество функций должно быть больше 0.")

        num_vars = int(input("Введите количество переменных в системе: "))
        if num_vars <= 0:
            raise ValueError("Количество переменных в системе должно быть больше 0.")

        functions = []
        for i in range(num_functions):
            print(f"Введите мономы для функции {i + 1}:")
            terms = list(map(int, input().strip().split()))
            
            if not all(term % 2 == 1 for term in terms):
                raise ValueError("Мономы представляются нечетными числами.")
            if not all(term < (1 << (num_vars + 1)) for term in terms):
                raise ValueError("Ошибка: моном содержит несуществующую переменную")

            functions.append(terms)

        system = BooleanFunctionSystem(num_functions, num_vars, max_terms=4)
        system.functions = functions
        return system

    @staticmethod
    def from_console_text():  # ввод системы функций (мономы представлены в виде x1...xi)
        num_functions = int(input("Введите количество функций: "))
        if num_functions <= 0:
            raise ValueError("Количество функций должно быть больше 0.")

        num_vars = int(input("Введите количество переменных в системе: "))
        if num_vars <= 0:
            raise ValueError("Количество переменных в системе должно быть больше 0.")

        functions = []
        for i in range(num_functions):
            while True:
                try:
                    print(f"\nФункция {i + 1}:")
                    function_str = input("Введите функцию: ").strip()
                
                    
                    monomials = set()
                    terms = [t.strip() for t in function_str.split('+') if t.strip()]
                
                    for term in terms:
                        if term == '1':
                            monomial = 1
                        elif term == '0':
                            monomial = 0
                        else:
                            monomial = 1
                            var_nums = []
                            current_num = []
                            for c in term:
                                if c == 'x':
                                    if current_num:
                                        var_nums.append(int(''.join(current_num)))
                                        current_num = []
                                elif c.isdigit():
                                    current_num.append(c)
                                else:
                                    raise ValueError(f"Некорректный символ в мономе: '{c}'")
                        
                            if current_num:
                                var_nums.append(int(''.join(current_num)))
                        
                            for var_num in var_nums:
                                if var_num < 1 or var_num > num_vars:
                                    raise ValueError(f"Переменная x{var_num} вне диапазона (1-{num_vars})")
                                monomial |= (1 << var_num)
                    
                        # XOR-логика для мономов
                        if monomial in monomials:
                            monomials.remove(monomial)
                        else:
                            monomials.add(monomial)
                
                    functions.append(list(monomials))
                    break
                
                except ValueError as e:
                    print(f"Ошибка: {e}. Пожалуйста, введите функцию снова.")

        max_terms = max(len(func) for func in functions)
        system = BooleanFunctionSystem(num_functions, num_vars, max_terms)
        system.functions = functions
        return system

    @staticmethod
    def from_functions(functions, num_vars=None):
        if not functions:
            raise ValueError("Список функций не может быть пустым")
    
        if num_vars is None:
            all_monomials = []
            for func in functions:
                all_monomials.extend(func)
        
            if not all_monomials:
                num_vars = 0
            else:
                max_monomial = max(all_monomials)
                if max_monomial == 1: 
                    num_vars = 0
                else:
                    num_vars = max_monomial.bit_length() - 1
     
        max_terms = max(len(func) for func in functions)
        num_functions = len(functions)
    
        system = BooleanFunctionSystem(num_functions, num_vars, max_terms, functions=functions)
    
        return system

    def substitute_variable(self, var_index, value):  # подстановка значения value в переменную var_index
        if var_index <= 0 or var_index > self.num_vars:
            raise ValueError("Нет такой переменной.")
        if value not in (0, 1):
            raise ValueError("Подставить можно только 0 или 1.")

        mask = 1 << var_index

        # Обработка каждой функции
        for function in self.functions:
            new_terms = set()

            # Обработка каждого монома в функции
            for monomial in function:
                # Если моном содержит переменную
                if monomial & mask:
                    if value == 0:
                        continue  # моном зануляется, не добавляем
                    else:
                        monomial &= ~mask  # убираем переменную из монома

                # Добавляем/удаляем моном (XOR-операция)
                if monomial not in new_terms:
                    new_terms.add(monomial)
                else:
                    new_terms.remove(monomial)

            # Если все мономы сократились - добавляем 0
            if len(new_terms) == 0:
                new_terms.add(0)

            # Обновляем функцию
            function[:] = list(new_terms)


    # вывод покрываемой матрицы (при выводе булевы значения преобразуются к 1/0)
    def _print_coverage_matrix(self, variables, monomials, matrix):
        print("\nПокрываемая матрица:")
    
        # Заголовок столбцов
        header = " " * 20
        for var in variables:
            header += f"{var:>6}"
        print(header)
    
        for i, (monom, row) in enumerate(zip(monomials, matrix)):
            monom_str = self.monomial_to_str(monom)
            # Преобразуем булевы значения в 1/0 для отображения
            row_display = ["1" if val else "0" for val in row]
            row_str = " ".join(f"{val:>6}" for val in row_display)
            print(f"{monom_str:<15} | {row_str}")

    # Реализация алгоритма поиска LP-множеств мощности 𝑛 с наибольшей нижней границей (алгоритм 1)
    def calculate_linearization_probability(self, n):
        if n <= 0 or n > self.num_vars - 2:
            raise ValueError(f"Размер подмножеств должен быть 1 <= n <= {self.num_vars-2}")
    
        X = set(range(1, self.num_vars + 1))
        p = {}
    
        # Генерируем все подмножества заранее
        
        for B_i in combinations(X, n):
            B_i_vars = tuple(f'x{var}' for var in B_i)
            
            # Создаем маску для переменных из B_i
            b_mask = 1
            for var in B_i:
                b_mask |= (1 << var)
            
            has_invalid_monomial = False
            excluded_monomials = set()
            
            for func in self.functions:
                for monom in func:
                    monom_vars = monom & ~1
                    
                    # Проверяем, сколько переменных монома НЕ входит в B_i
                    external_vars_mask = monom_vars & ~b_mask
                    
                    # Считаем количество таких переменных (X\B_i)
                    num_external = calculate_weight(external_vars_mask)
                    
                    if num_external > 1:
                        # Если есть более одной внешней переменной
                        # Проверяем, есть ли хоть одна переменная из B_i в мономе
                        if (monom_vars & b_mask) == 0:
                            has_invalid_monomial = True
                            break
                        excluded_monomials.add(monom)
                
                if has_invalid_monomial:
                    break
            
            if has_invalid_monomial:
                p[B_i_vars] = 0.0
            elif not excluded_monomials:
                p[B_i_vars] = 1.0
            else:
                # Построение матрицы покрытия
                matrix = []
                for monom in excluded_monomials:
                    row = [bool(monom & (1 << var)) for var in B_i]
                    matrix.append(row)
                
                #print(f"\nПодмножество {', '.join(B_i_vars)}:")
                #self._print_coverage_matrix(B_i_vars, excluded_monomials, matrix)
                
                # Расчет вероятности
                k, cv = self._find_greedy_cover(matrix, B_i_vars)
                prob = 1.0 / (2 ** k)
                print(f"Покрывающее множество: {cv}")
                p[B_i_vars] = prob
        
        # Вывод результатов
        sets = []
        print("\nИТОГОВЫЕ ВЕРОЯТНОСТИ:")
        max_prob = max(p.values()) if p else 0
        for vars, prob in sorted(p.items(), key=lambda x: (-x[1], x[0])):
            star = " *" if prob == max_prob else ""
            if prob == max_prob: 
                sets.append(vars)
            print(f"{', '.join(vars)}: {prob:.6f}{star}")
        print(f"Лучшие множества: {sets}")
        return p, sets

    def monomial_to_str(self, monomial):
        if monomial == 0:
            return "0"
        if monomial == 1:
            return "1"
        return ''.join(f'x{i}' for i in range(1, self.num_vars+1) 
                      if monomial & (1 << i))

    # жадный алгоритм поиска покрытия матрицы
    def _find_greedy_cover(self, matrix, variables):
        if not matrix or not variables:
            return 0, []
    
        rows = len(matrix)
        cols = len(variables)
        covered = [False] * rows
        selected = []
    
        while True:
            # Находим столбец с максимальным покрытием
            best_col = None
            max_cover = 0
        
            for col in range(cols):
                if col not in selected:
                    cover_count = sum(
                        1 for row in range(rows) 
                        if not covered[row] and matrix[row][col]
                    )
            
                    if cover_count > max_cover:
                        max_cover = cover_count
                        best_col = col
        
            # Если не нашли столбец или он ничего не покрывает
            if best_col is None or max_cover == 0:
                break
            
            selected.append(best_col)
        
            # Помечаем покрытые строки
            for row in range(rows):
                if matrix[row][best_col]:
                    covered[row] = True
    
        # Проверяем, все ли строки покрыты
        if not all(covered):
            return 0, []
    
        return len(selected), [variables[col] for col in selected]

    # Поиск минимального покрытия (полный перебор)
    def _find_exact_cover(self, matrix, variables):
        if not matrix or not variables:
            return 0, []
    
        cols = len(variables)
        rows = len(matrix)
    
        # Перебираем размер покрытия от 1 до всех столбцов
        for k in range(1, cols + 1):
            # Проверяем все комбинации из k столбцов
            for cols_combo in combinations(range(cols), k):
                # Проверяем, покрывают ли выбранные столбцы все строки
                covered_all = True
                for row in range(rows):
                    if not any(matrix[row][col] for col in cols_combo):
                        covered_all = False
                        break
            
                if covered_all:
                    # Возвращаем мощность и список переменных
                    return k, [variables[col] for col in cols_combo]
    
        # Если не нашли (все строки нулевые)
        return len(cols), variables.copy()

    # алгортим полного перебора (алгоритм 3)
    def brute_force_linearization_check(self, subset_size):
    
        X = list(range(1, self.num_vars + 1))
        results = {}

        # Предварительно создаем все возможные комбинации значений
        all_value_combinations = list(product([0, 1], repeat=subset_size))
    
        for B_i in combinations(X, subset_size):
            B_i_list = list(B_i)
            B_i_vars = tuple(f'x{var}' for var in B_i_list)
            total_cases = 1 << subset_size
            success_cases = 0
        
            # Используем предсозданные комбинации значений
            for values in all_value_combinations:
                temp_system = self.copy()
            
                # Подставляем значения переменных
                for i in range(subset_size):
                    temp_system.substitute_variable(B_i_list[i], values[i])
            
                # Проверяем, стала ли система линейной
                if temp_system.is_linear():
                    success_cases += 1
        
            # Вычисляем вероятность линеаризации
            probability = success_cases / total_cases
            results[B_i_vars] = probability
        
            # Сохраняем исходный вывод
            print(f"Подмножество {B_i_vars}: "
                  f"успешных случаев = {success_cases}/{total_cases}, "
                  f"вероятность = {probability:.4f}")
    
        # Находим подмножества с максимальной вероятностью
        max_prob = max(results.values()) if results else 0
        best_subsets = [vars for vars, prob in results.items() if prob == max_prob]
    
        print("\nРезультаты:")
        print(f"Максимальная вероятность линеаризации: {max_prob:.4f}")
        print("Лучшие подмножества:", ", ".join(map(str, best_subsets)))
    
        return results, best_subsets

    # Оптимизация алгоритма поиска LP-множеств мощности 𝑛 с наибольшей нижней границей (алгоритм 3)
    def calculate_linearization_probability_optimized(self, n):
        if n <= 0 or n > self.num_vars - 2:
            raise ValueError(f"Размер подмножеств должен быть 1 <= n <= {self.num_vars-2}")
    
        X = set(range(1, self.num_vars + 1))
        p = {}
    
        excluded_sets = set()
        max_degree_for_filter = self.num_vars - n
    
        # Предварительный обход системы
        for func in self.functions:
            for monom in func:
                degree = calculate_weight(monom) - 1
            
                # Степень монома не превосходит n-k
                if 2 <= degree <= max_degree_for_filter:
                    monom_vars_set = set()
                    for i in range(1, self.num_vars + 1):
                        if monom & (1 << i):
                            monom_vars_set.add(i)
                
                    # Все множества, не содержащие переменных монома, дают p=0
                    for bad_B_i in combinations(X - monom_vars_set, n):
                        excluded_sets.add(tuple(bad_B_i))
    
        # Множества для анализа
        all_sets = set(combinations(X, n))
        potential_sets = all_sets - excluded_sets
    
        print(f"Всего множеств: {len(all_sets)}, исключено: {len(excluded_sets)}, осталось: {len(potential_sets)}")
    
        for B_i in potential_sets:
            B_i_vars = tuple(f'x{var}' for var in B_i)
            
            b_mask = 1
            for var in B_i:
                b_mask |= (1 << var)
            
            excluded_monomials = set()
            
            # Поиск покрываемых мономов (после фильтрации все мономы имеют хотя бы одну переменную из B_i)
            for func in self.functions:
                for monom in func:
                    monom_vars = monom & ~1
                    
                    # Проверяем, сколько переменных монома не входит в B_i
                    external_vars_mask = monom_vars & ~b_mask
                    num_external = calculate_weight(external_vars_mask)
                    
                    # Если есть более одной внешней переменной, моном требует покрытия
                    if num_external > 1: 
                        excluded_monomials.add(monom)
    
            if not excluded_monomials:
                p[B_i_vars] = 1.0
            else:
                # Построение матрицы покрытия
                matrix = []
                for monom in excluded_monomials:
                    row = [bool(monom & (1 << var)) for var in B_i]
                    matrix.append(row)
        
                #print(f"\nПодмножество {', '.join(B_i_vars)}:")
                #self._print_coverage_matrix(B_i_vars, excluded_monomials, matrix)
        
                # Расчет вероятности
                k, cv = self._find_greedy_cover(matrix, B_i_vars)
                prob = 1.0 / (2 ** k) if k > 0 else 0.0
                print(f"Покрывающее множество: {cv}")
                p[B_i_vars] = prob

        # Для исключенных множеств проставляем 0.0
        for B_i in excluded_sets:
            B_i_vars = tuple(f'x{var}' for var in B_i)
            p[B_i_vars] = 0.0

        sets = []
        print("\nИТОГОВЫЕ ВЕРОЯТНОСТИ:")
        max_prob = max(p.values()) if p else 0
        for vars, prob in sorted(p.items(), key=lambda x: (-x[1], x[0])):
            star = " *" if prob == max_prob else ""
            if prob == max_prob: 
                sets.append(vars)
            print(f"{', '.join(vars)}: {prob:.6f}{star}")
        print(f"Лучшие множества: {sets}")
        return p, sets, len(excluded_sets)

    # Алгоритм поиска LP-множеств мощности 𝑛 с наибольшей нижней границей с уточнениями
    def calculate_refined_probability(self, n, max_iter=3):
        X = set(range(1, self.num_vars + 1))
        results = {}
        global_max_prob = 0.0
        best_subsets = []

        for B_i in combinations(X, n):
            B_i_vars = tuple(f'x{var}' for var in B_i)
            print(f"\n\n{'='*60}\nАнализ множества: {', '.join(B_i_vars)}\n{'='*60}")
            b_mask = 1
            for var in B_i:
                b_mask |= (1 << var)
            
            # Проверка на недопустимые мономы
            has_invalid_monomial = False
            for func in self.functions:
                for monom in func:
                    monom_vars = monom & ~1
                    
                    # Проверяем, сколько переменных монома не входит в B_i
                    external_vars_mask = monom_vars & ~b_mask
                    num_external = calculate_weight(external_vars_mask)
                    
                    if num_external > 1 and (monom_vars & b_mask) == 0:
                        has_invalid_monomial = True
                        print(f"Недопустимый моном: {self.monomial_to_str(monom)}")
                        break
                if has_invalid_monomial:
                    break

            if has_invalid_monomial:
                results[B_i_vars] = 0.0
                print("=> p=0.0 (найдены непокрываемые мономы)")
                continue

            func_copy = [func.copy() for func in self.functions]
            total_prob = 0.0
            used_vars = set()
            used_vars_mask = 1  # Маска для использованных переменных

            for iteration in range(max_iter):
                # Сбор и упрощение мономов с использованием масок
                current_monomials = set()
                for func in func_copy:
                    simplified_func = set()
                    for monom in func:
                        # Упрощаем моном, убирая использованные переменные
                        simplified_monom = (monom & ~used_vars_mask)|1
                        if simplified_monom != 1:
                            simplified_func.add(simplified_monom)
                    current_monomials.update(simplified_func)

                # Поиск проблемных мономов для текущей итерации
                excluded_monomials = set()
                for monom in current_monomials:
                    monom_vars = monom & ~1
                    
                    # Проверяем, сколько переменных из X\B_i
                    # Создаем маску для активных переменных B_i
                    active_b_mask = b_mask & ~used_vars_mask
                    
                    # Переменные монома, которые не входят в активные переменные B_i
                    external_vars_mask = monom_vars & ~active_b_mask
                    num_external = calculate_weight(external_vars_mask)
                    
                    if num_external > 1:
                        excluded_monomials.add(monom)

                if not excluded_monomials:
                    if iteration == 0:
                        total_prob = 1.0
                        print("Нет проблемных мономов -> p=1.0")
                    break

                # Построение матрицы покрытия 
                active_vars = []
                active_vars_indices = []
                for var in B_i:
                    var_mask = 1 << var
                    if var_mask & ~used_vars_mask:  # Если переменная еще не использована
                        active_vars.append(f'x{var}')
                        active_vars_indices.append(var)
                
                matrix = [
                    [bool(monom & (1 << var_idx)) for var_idx in active_vars_indices]
                    for monom in excluded_monomials
                ]

                #print(f"\nПодмножество {', '.join(B_i_vars)}:")
                #self._print_coverage_matrix(active_vars, excluded_monomials, matrix)

                # Поиск покрытия
                k, cover_vars = self._find_greedy_cover(matrix, active_vars)
                if k == 0:
                    print("Не удалось найти покрытие - завершение итераций")
                    break

                # Обновляем использованные переменные
                used_vars.update(cover_vars)
                for var_str in cover_vars:
                    var_idx = int(var_str[1:])
                    used_vars_mask |= (1 << var_idx)
                
                current_k = len(used_vars)
                prob_increment = 1.0 / (2 ** current_k)
                total_prob += prob_increment

                print(f"\nНайдено покрытие: {cover_vars} (k={k})")
                print(f"Добавлена вероятность: 1/2^{current_k} = {prob_increment:.6f}")
                print(f"Текущая суммарная вероятность: {total_prob:.6f}")

            results[B_i_vars] = total_prob

            # Обновление лучших подмножеств
            if total_prob > global_max_prob:
                global_max_prob = total_prob
                best_subsets = [B_i_vars]
            elif total_prob == global_max_prob:
                best_subsets.append(B_i_vars)

            print(f"\nИтог для {', '.join(B_i_vars)}: p = {total_prob:.6f}")

        print("\n" + "="*60)
        print(f"МАКСИМАЛЬНАЯ ВЕРОЯТНОСТЬ: {global_max_prob:.6f}")
        print("ЛУЧШИЕ ПОДМНОЖЕСТВА:")
        for subset in best_subsets:
            print(f"- {', '.join(subset)}")

        return results, best_subsets
    
    # Алгоритм поиска LP-множеств мощности 𝑛 с наибольшей нижней границей с уточнениями
    def calculate_refined_probability_optimized(self, n, max_iter=3):
        X = set(range(1, self.num_vars + 1))
        results = {}
        global_max_prob = 0.0
        best_subsets = []
    
        excluded_sets = set()
        max_degree_for_filter = self.num_vars - n
    
        # Предварительный обход системы
        for func in self.functions:
            for monom in func:
                degree = calculate_weight(monom) - 1
            
                if 2 <= degree <= max_degree_for_filter:
                    monom_vars_set = set()
                    for i in range(1, self.num_vars + 1):
                        if monom & (1 << i):
                            monom_vars_set.add(i)
                    
                    # Все множества, не содержащие переменных монома, дают p=0
                    for bad_B_i in combinations(X - monom_vars_set, n):
                        excluded_sets.add(tuple(sorted(bad_B_i)))
    
        # Множества для анализа
        all_sets = set(combinations(X, n))
        potential_sets = all_sets - excluded_sets
    
        print(f"Всего множеств: {len(all_sets)}, исключено: {len(excluded_sets)}, осталось: {len(potential_sets)}")
    
        for B_i in potential_sets:
            B_i_vars = tuple(f'x{var}' for var in B_i)
            print(f"\n\n{'='*60}\nАнализ множества: {', '.join(B_i_vars)}\n{'='*60}")

            b_mask = 1
            for var in B_i:
                b_mask |= (1 << var)
            
            func_copy = [func.copy() for func in self.functions]
            total_prob = 0.0
            used_vars = set()
            used_vars_mask = 0

            for iteration in range(max_iter):
                # Сбор и упрощение мономов
                current_monomials = set()
                for func in func_copy:
                    simplified_func = set()
                    for monom in func:
                        # Упрощаем моном, убирая использованные переменные
                        simplified_monom = (monom & ~used_vars_mask)|1
                        if simplified_monom != 1:
                            simplified_func.add(simplified_monom)
                    current_monomials.update(simplified_func)

                # Поиск проблемных мономов
                excluded_monomials = set()
                for monom in current_monomials:
                    monom_vars = monom & ~1
                    active_b_mask = b_mask & ~used_vars_mask
                    external_vars_mask = monom_vars & ~active_b_mask
                    num_external = calculate_weight(external_vars_mask)
                    
                    # После предфильтрации все мономы гарантированно имеют хотя бы одну переменную из B_i
                    if num_external > 1:
                        excluded_monomials.add(monom)

                if not excluded_monomials:
                    if iteration == 0:
                        total_prob = 1.0
                        print("Нет проблемных мономов -> p=1.0")
                    break

                # Построение матрицы покрытия
                active_vars = []
                active_vars_indices = []
                for var in B_i:
                    var_mask = 1 << var
                    if var_mask & ~used_vars_mask:
                        active_vars.append(f'x{var}')
                        active_vars_indices.append(var)
                
                matrix = [
                    [bool(monom & (1 << var_idx)) for var_idx in active_vars_indices]
                    for monom in excluded_monomials
                ]

                #print(f"\nПодмножество {', '.join(B_i_vars)}:")
                #self._print_coverage_matrix(active_vars, excluded_monomials, matrix)

                # Поиск покрытия
                k, cover_vars = self._find_greedy_cover(matrix, active_vars)
                if k == 0:
                    print("Не удалось найти покрытие - завершение итераций")
                    break

                # Обновляем использованные переменные
                used_vars.update(cover_vars)
                for var_str in cover_vars:
                    var_idx = int(var_str[1:])
                    used_vars_mask |= (1 << var_idx)
                
                current_k = len(used_vars)
                prob_increment = 1.0 / (2 ** current_k)
                total_prob += prob_increment

                print(f"\nНайдено покрытие: {cover_vars} (k={k})")
                print(f"Добавлена вероятность: 1/2^{current_k} = {prob_increment:.6f}")
                print(f"Текущая суммарная вероятность: {total_prob:.6f}")

            results[B_i_vars] = total_prob

            # Обновление лучших подмножеств
            if total_prob > global_max_prob:
                global_max_prob = total_prob
                best_subsets = [B_i_vars]
            elif total_prob == global_max_prob:
                best_subsets.append(B_i_vars)

            print(f"\nИтог для {', '.join(B_i_vars)}: p = {total_prob:.6f}")

        # Для исключенных множеств проставляем 0.0
        for B_i in excluded_sets:
            B_i_vars = tuple(f'x{var}' for var in B_i)
            results[B_i_vars] = 0.0

        print(f"\n{'='*60}")
        print(f"МАКСИМАЛЬНАЯ ВЕРОЯТНОСТЬ: {global_max_prob:.6f}")
        print("ЛУЧШИЕ ПОДМНОЖЕСТВА:")
        for subset in best_subsets:
            print(f"- {', '.join(subset)}")

        return results, best_subsets

if __name__ == "__main__":
    system = BooleanFunctionSystem.from_console_text()
    system.pretty_print()