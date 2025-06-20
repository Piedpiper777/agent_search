"""
Agent Search System - 工具特定多目标动态规划+强化学习+LLM智能
每个工具使用特定的目标维度，实现更精确的多目标优化
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import random

def AgentAction(value: str):
    """模拟AgentAction枚举"""
    return value

# ===== 1. 目标类型定义 =====
class ObjectiveType(Enum):
    """目标类型枚举"""
    # 通用目标
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency" 
    COMPLETENESS = "completeness"
    COST = "cost"
    
    # 分解特定目标
    CLARITY = "clarity"              # 分解清晰度
    COVERAGE = "coverage"            # 问题覆盖度
    ACTIONABILITY = "actionability"  # 可操作性
    
    # 搜索特定目标
    RELEVANCE = "relevance"          # 相关性
    AUTHORITY = "authority"          # 权威性
    NOVELTY = "novelty"             # 新颖性
    RECALL = "recall"               # 召回率
    PRECISION = "precision"         # 精确率
    
    # 生成特定目标
    COHERENCE = "coherence"         # 连贯性
    FLUENCY = "fluency"            # 流畅性
    FACTUALITY = "factuality"      # 事实准确性
    COMPREHENSIVENESS = "comprehensiveness"  # 全面性

# ===== 2. 增强的核心数据结构 =====
@dataclass
class QuantitativeState:
    """量化状态表示"""
    stage: int                      # 决策阶段
    facts_completeness: float       # 事实完整度 [0,1]
    query_complexity: float         # 查询复杂度 [0,1]
    search_exhaustion: float        # 搜索穷尽度 [0,1]
    time_elapsed: float             # 时间消耗比例 [0,1]
    confidence_level: float         # 当前置信度 [0,1]
    
    def to_key(self) -> str:
        """转换为状态键（用于记忆化）"""
        return f"{self.stage}_{self.facts_completeness:.1f}_{self.query_complexity:.1f}_{self.time_elapsed:.1f}_{self.search_exhaustion:.1f}_{self.confidence_level:.1f}"
    
    def to_vector(self) -> np.ndarray:
        """转换为向量形式"""
        return np.array([
            self.facts_completeness,
            self.query_complexity, 
            self.search_exhaustion,
            self.time_elapsed,
            self.confidence_level
        ])

@dataclass
class ParetoSolution:
    """帕累托最优解"""
    action: AgentAction
    state: QuantitativeState
    objectives: np.ndarray               # 目标向量（维度可变）
    objective_types: List[ObjectiveType] # 目标类型列表
    policy_path: List[AgentAction]       # 从当前状态到终点的动作序列
    expected_reward: float               # 期望总奖励
    confidence: float                    # 解的置信度
    tool_specific_score: float = 0.0     # 工具特定评分

# ===== 3. 工具特定目标管理器 =====
class ToolSpecificObjectives:
    """工具特定目标管理器"""
    
    def __init__(self):
        # 每个工具的特定目标
        self.tool_objectives = {
            AgentAction.DECOMPOSE: [
                ObjectiveType.CLARITY,
                ObjectiveType.COVERAGE, 
                ObjectiveType.ACTIONABILITY,
                ObjectiveType.COST
            ],
            AgentAction.SEARCH_KG: [
                ObjectiveType.RELEVANCE,
                ObjectiveType.AUTHORITY,
                ObjectiveType.RECALL,
                ObjectiveType.COST
            ],
            AgentAction.SEARCH_VECTOR: [
                ObjectiveType.RELEVANCE,
                ObjectiveType.NOVELTY,
                ObjectiveType.PRECISION,
                ObjectiveType.COST
            ],
            AgentAction.GENERATE: [
                ObjectiveType.FACTUALITY,
                ObjectiveType.COHERENCE,
                ObjectiveType.COMPREHENSIVENESS,
                ObjectiveType.FLUENCY
            ],
            AgentAction.REFINE: [
                ObjectiveType.ACCURACY,
                ObjectiveType.PRECISION,
                ObjectiveType.COHERENCE,
                ObjectiveType.COST
            ]
        }
        
        # 目标计算器
        self.objective_calculators = self._initialize_calculators()
        
        # 动态权重管理器
        self.weight_manager = AdaptiveObjectiveWeighting()
    
    def get_objectives_for_tool(self, action: AgentAction) -> List[ObjectiveType]:
        """获取工具的特定目标"""
        return self.tool_objectives.get(action, [
            ObjectiveType.ACCURACY,
            ObjectiveType.EFFICIENCY,
            ObjectiveType.COMPLETENESS,
            ObjectiveType.COST
        ]) #如果没有特定目标，则返回通用目标
    
    def calculate_tool_objectives(self, 
                                state: QuantitativeState, 
                                action: AgentAction) -> Tuple[np.ndarray, List[ObjectiveType]]:
        """计算工具特定的目标向量"""
        objectives = self.get_objectives_for_tool(action)
        objective_values = []
        
        for obj_type in objectives:
            calculator = self.objective_calculators[obj_type]
            value = calculator(state, action)
            objective_values.append(value)
        
        return np.array(objective_values), objectives
    
    def _initialize_calculators(self) -> Dict:
        """初始化目标计算器"""
        return {
            # 分解目标计算器
            ObjectiveType.CLARITY: self._calculate_clarity,
            ObjectiveType.COVERAGE: self._calculate_coverage,
            ObjectiveType.ACTIONABILITY: self._calculate_actionability,
            
            # 搜索目标计算器
            ObjectiveType.RELEVANCE: self._calculate_relevance,
            ObjectiveType.AUTHORITY: self._calculate_authority,
            ObjectiveType.NOVELTY: self._calculate_novelty,
            ObjectiveType.RECALL: self._calculate_recall,
            ObjectiveType.PRECISION: self._calculate_precision,
            
            # 生成目标计算器
            ObjectiveType.FACTUALITY: self._calculate_factuality,
            ObjectiveType.COHERENCE: self._calculate_coherence,
            ObjectiveType.COMPREHENSIVENESS: self._calculate_comprehensiveness,
            ObjectiveType.FLUENCY: self._calculate_fluency,
            
            # 通用目标计算器
            ObjectiveType.ACCURACY: self._calculate_accuracy,
            ObjectiveType.EFFICIENCY: self._calculate_efficiency,
            ObjectiveType.COMPLETENESS: self._calculate_completeness,
            ObjectiveType.COST: self._calculate_cost
        }
    
    # ===== 分解目标计算器 =====
    def _calculate_clarity(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算分解清晰度"""
        if action != AgentAction.DECOMPOSE:
            return 0.0
        
        base_clarity = 0.6
        complexity_bonus = 0.3 * state.query_complexity  # 复杂查询分解收益更大
        confidence_bonus = 0.1 * state.confidence_level
        
        return min(1.0, base_clarity + complexity_bonus + confidence_bonus)
    
    def _calculate_coverage(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算问题覆盖度"""
        if action != AgentAction.DECOMPOSE:
            return 0.0
        
        coverage = 0.5 + 0.4 * state.query_complexity
        
        # 如果已经分解过，覆盖度递减
        if state.stage > 1:
            coverage *= 0.8
            
        return min(1.0, coverage)
    
    def _calculate_actionability(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算可操作性"""
        if action != AgentAction.DECOMPOSE:
            return 0.0
        
        actionability = 0.7
        
        # 时间压力下，简单可操作的分解更有价值
        if state.time_elapsed > 0.5:
            actionability += 0.2 * (1 - state.time_elapsed)
        
        return min(1.0, actionability)
    
    # ===== 搜索目标计算器 =====
    def _calculate_relevance(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算搜索相关性"""
        if action not in [AgentAction.SEARCH_KG, AgentAction.SEARCH_VECTOR]:
            return 0.0
        
        base_relevance = 0.6
        
        if action == AgentAction.SEARCH_KG:
            # KG搜索对结构化查询相关性更高
            structural_bonus = 0.2 * (1 - state.query_complexity)
            return min(1.0, base_relevance + structural_bonus)
        else:  # SEARCH_VECTOR
            # 向量搜索对复杂语义查询相关性更高
            semantic_bonus = 0.3 * state.query_complexity
            return min(1.0, base_relevance + semantic_bonus)
    
    def _calculate_authority(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算信息权威性"""
        if action != AgentAction.SEARCH_KG:
            return 0.0
        
        authority = 0.8
        exhaustion_penalty = 0.2 * state.search_exhaustion
        
        return max(0.1, authority - exhaustion_penalty)
    
    def _calculate_novelty(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算信息新颖性"""
        if action != AgentAction.SEARCH_VECTOR:
            return 0.0
        
        base_novelty = 0.5
        novelty_potential = 0.4 * (1 - state.search_exhaustion)
        completeness_factor = 0.2 * (1 - state.facts_completeness)
        
        return min(1.0, base_novelty + novelty_potential + completeness_factor)
    
    def _calculate_recall(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算召回率"""
        if action != AgentAction.SEARCH_KG:
            return 0.0
        
        # KG搜索召回率通常较高，但随搜索深度递减
        recall = 0.8 - 0.3 * state.search_exhaustion
        return max(0.2, recall)
    
    def _calculate_precision(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算精确率"""
        if action not in [AgentAction.SEARCH_VECTOR, AgentAction.REFINE]:
            return 0.0
        
        if action == AgentAction.SEARCH_VECTOR:
            # 向量搜索精确率中等，但语义匹配好
            precision = 0.6 + 0.2 * (1 - state.query_complexity)
        else:  # REFINE
            # 精化操作提高精确率
            precision = 0.7 + 0.2 * state.confidence_level
        
        return min(1.0, precision)
    
    # ===== 生成目标计算器 =====
    def _calculate_factuality(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算事实准确性"""
        if action != AgentAction.GENERATE:
            return 0.0
        
        factuality = 0.3 + 0.6 * state.facts_completeness
        confidence_bonus = 0.1 * state.confidence_level
        
        return min(1.0, factuality + confidence_bonus)
    
    def _calculate_coherence(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算生成连贯性"""
        if action not in [AgentAction.GENERATE, AgentAction.REFINE]:
            return 0.0
        
        coherence = 0.8
        complexity_penalty = 0.2 * state.query_complexity
        
        return max(0.3, coherence - complexity_penalty)
    
    def _calculate_comprehensiveness(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算全面性"""
        if action != AgentAction.GENERATE:
            return 0.0
        
        return 0.4 + 0.5 * state.facts_completeness
    
    def _calculate_fluency(self, state: QuantitativeState, action: AgentAction) -> float:
        """计算流畅性"""
        if action != AgentAction.GENERATE:
            return 0.0
        
        # LLM生成通常流畅性很好
        return 0.9
    
    # ===== 通用目标计算器（保持兼容性） =====
    def _calculate_accuracy(self, state: QuantitativeState, action: AgentAction) -> float:
        """通用准确性计算"""
        if action == AgentAction.DECOMPOSE:
            return self._calculate_clarity(state, action)
        elif action in [AgentAction.SEARCH_KG, AgentAction.SEARCH_VECTOR]:
            return self._calculate_relevance(state, action)
        elif action == AgentAction.GENERATE:
            return self._calculate_factuality(state, action)
        else:
            return 0.5
    
    def _calculate_efficiency(self, state: QuantitativeState, action: AgentAction) -> float:
        """通用效率计算"""
        base_efficiency = {
            AgentAction.DECOMPOSE: 0.8,
            AgentAction.SEARCH_KG: 0.5,
            AgentAction.SEARCH_VECTOR: 0.7,
            AgentAction.GENERATE: 0.9,
            AgentAction.REFINE: 0.4
        }.get(action, 0.5)
        
        time_penalty = 0.3 * state.time_elapsed
        return max(0.1, base_efficiency - time_penalty)
    
    def _calculate_completeness(self, state: QuantitativeState, action: AgentAction) -> float:
        """通用完整性计算"""
        if action in [AgentAction.SEARCH_KG, AgentAction.SEARCH_VECTOR]:
            return 0.6 * (1 - state.search_exhaustion)
        elif action == AgentAction.DECOMPOSE:
            return self._calculate_coverage(state, action)
        else:
            return 0.1
    
    def _calculate_cost(self, state: QuantitativeState, action: AgentAction) -> float:
        """通用成本计算（负值表示成本）"""
        base_costs = {
            AgentAction.DECOMPOSE: -0.1,
            AgentAction.SEARCH_KG: -0.2,
            AgentAction.SEARCH_VECTOR: -0.15,
            AgentAction.GENERATE: -0.3,
            AgentAction.REFINE: -0.2
        }
        
        base_cost = base_costs.get(action, -0.1)
        
        # 复杂度影响成本
        complexity_factor = 1 + 0.5 * state.query_complexity
        
        return base_cost * complexity_factor

# ===== 4. 自适应权重管理器 =====
class AdaptiveObjectiveWeighting:
    """自适应目标权重管理器"""
    
    def __init__(self):
        # 工具特定的基础权重
        self.base_weights = {
            AgentAction.DECOMPOSE: {
                ObjectiveType.CLARITY: 0.4,
                ObjectiveType.COVERAGE: 0.3,
                ObjectiveType.ACTIONABILITY: 0.2,
                ObjectiveType.COST: 0.1
            },
            AgentAction.SEARCH_KG: {
                ObjectiveType.RELEVANCE: 0.35,
                ObjectiveType.AUTHORITY: 0.35,
                ObjectiveType.RECALL: 0.2,
                ObjectiveType.COST: 0.1
            },
            AgentAction.SEARCH_VECTOR: {
                ObjectiveType.RELEVANCE: 0.4,
                ObjectiveType.NOVELTY: 0.3,
                ObjectiveType.PRECISION: 0.2,
                ObjectiveType.COST: 0.1
            },
            AgentAction.GENERATE: {
                ObjectiveType.FACTUALITY: 0.4,
                ObjectiveType.COHERENCE: 0.25,
                ObjectiveType.COMPREHENSIVENESS: 0.25,
                ObjectiveType.FLUENCY: 0.1
            }
        }
        
        self.adaptation_history = defaultdict(list)
        self.user_preferences = {}
    
    def get_dynamic_weights(self, 
                          action: AgentAction, 
                          state: QuantitativeState,
                          user_preference: Optional[Dict[str, float]] = None) -> Dict[ObjectiveType, float]:
        """获取动态调整的权重"""
        
        base_weights = self.base_weights.get(action, {}).copy()
        
        # 基于状态的权重调整
        if action == AgentAction.DECOMPOSE:
            if state.query_complexity > 0.7:
                # 复杂查询更重视覆盖度
                base_weights[ObjectiveType.COVERAGE] = base_weights.get(ObjectiveType.COVERAGE, 0.3) + 0.1
                base_weights[ObjectiveType.CLARITY] = base_weights.get(ObjectiveType.CLARITY, 0.4) - 0.1
        
        elif action == AgentAction.SEARCH_KG:
            if state.time_elapsed > 0.6:
                # 时间紧迫时更重视成本
                base_weights[ObjectiveType.COST] = base_weights.get(ObjectiveType.COST, 0.1) + 0.1
                base_weights[ObjectiveType.RECALL] = base_weights.get(ObjectiveType.RECALL, 0.2) - 0.1
        
        elif action == AgentAction.SEARCH_VECTOR:
            if state.facts_completeness < 0.4:
                # 事实不足时更重视新颖性
                base_weights[ObjectiveType.NOVELTY] = base_weights.get(ObjectiveType.NOVELTY, 0.3) + 0.15
                base_weights[ObjectiveType.RELEVANCE] = base_weights.get(ObjectiveType.RELEVANCE, 0.4) - 0.15
        
        # 融入用户偏好
        if user_preference:
            base_weights = self._incorporate_user_preference(base_weights, user_preference)
        
        # 归一化权重
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            for key in base_weights:
                base_weights[key] /= total_weight
        
        return base_weights
    
    def _incorporate_user_preference(self, weights: Dict, preferences: Dict) -> Dict:
        """融入用户偏好"""
        # 简化的偏好融入策略
        for obj_name, pref_weight in preferences.items():
            obj_type = ObjectiveType(obj_name) if obj_name in [e.value for e in ObjectiveType] else None
            if obj_type and obj_type in weights:
                weights[obj_type] *= (1 + pref_weight)
        
        return weights

# ===== 5. 增强的多目标动态规划求解器 =====
class EnhancedMultiObjectiveDynamicProgramming:
    """增强的多目标动态规划求解器 - 支持工具特定目标"""
    
    def __init__(self):
        # 帕累托最优解记忆表: state_key -> List[ParetoSolution]
        self.pareto_memo = {}
        
        # 核心组件
        self.state_transitions = StateTransitionModel()
        self.tool_objectives = ToolSpecificObjectives()
        self.feasibility_checker = ActionFeasibilityChecker()
        
        # 算法参数
        self.max_stages = 10
        self.gamma = 0.9
        
        # 性能监控
        self.solver_stats = {
            'memo_hits': 0,   # 缓存命中次数
            'memo_misses': 0, # 缓存未命中次数
            'pareto_solutions_generated': 0, # 生成的帕累托最优解数量
            'objective_dimensions_used': defaultdict(int) # 目标维度使用统计
        }
    
    def solve(self, initial_state: QuantitativeState) -> List[ParetoSolution]:
        """主求解函数 - 支持动态目标维度"""
        print(f"开始增强多目标动态规划求解，初始状态: {initial_state}")
        
        # 清空统计
        self.solver_stats = {
            'memo_hits': 0,
            'memo_misses': 0,
            'pareto_solutions_generated': 0,
            'objective_dimensions_used': defaultdict(int)
        }
        
        # 递归求解
        pareto_solutions = self._solve_recursive(initial_state, self.max_stages)
        
        print(f"求解完成: 找到 {len(pareto_solutions)} 个帕累托最优解")
        print(f"缓存命中率: {self.solver_stats['memo_hits']}/{self.solver_stats['memo_hits'] + self.solver_stats['memo_misses']}")
        
        return pareto_solutions
    
    def _solve_recursive(self, state: QuantitativeState, stages_remaining: int) -> List[ParetoSolution]:
        """支持工具特定目标的递归求解"""
        state_key = state.to_key()
        
        # Step 1: 记忆化检查
        if state_key in self.pareto_memo:
            self.solver_stats['memo_hits'] += 1
            return self.pareto_memo[state_key]
        
        self.solver_stats['memo_misses'] += 1
        
        # Step 2: 边界条件检查
        if stages_remaining == 0 or self._is_terminal_state(state):
            terminal_solutions = self._get_terminal_solutions(state)
            self.pareto_memo[state_key] = terminal_solutions
            return terminal_solutions
        
        # Step 3: 获取可行动作
        feasible_actions = self._get_feasible_actions(state)
        if not feasible_actions:
            return self._get_terminal_solutions(state)
        
        # Step 4: 遍历所有可行动作，收集候选解
        all_candidate_solutions = []
        
        for action in feasible_actions:
            # 计算该工具的特定目标向量
            immediate_rewards, objective_types = self.tool_objectives.calculate_tool_objectives(state, action)
            
            # 统计目标维度使用
            for obj_type in objective_types:
                self.solver_stats['objective_dimensions_used'][obj_type.value] += 1
            
            # 获取下一状态分布
            next_state_distribution = self.state_transitions.get_next_states(state, action)
            
            # 对每个可能的下一状态
            for next_state, transition_prob in next_state_distribution:
                # 递归求解下一状态
                future_solutions = self._solve_recursive(next_state, stages_remaining - 1)
                
                # 组合不同维度的目标向量
                for future_solution in future_solutions:
                    combined_objectives, combined_types = self._combine_heterogeneous_objectives(
                        immediate_rewards,
                        objective_types,
                        future_solution.objectives,
                        future_solution.objective_types,
                        transition_prob
                    )
                    
                    # 计算工具特定评分
                    tool_score = self._calculate_tool_specific_score(
                        action, combined_objectives, combined_types, state
                    )
                    
                    candidate_solution = ParetoSolution(
                        action=action,
                        state=state,
                        objectives=combined_objectives,
                        objective_types=combined_types,
                        policy_path=[action] + future_solution.policy_path,
                        expected_reward=np.sum(combined_objectives),
                        confidence=transition_prob * future_solution.confidence,
                        tool_specific_score=tool_score
                    )
                    
                    all_candidate_solutions.append(candidate_solution)
        
        # Step 5: 多维度帕累托筛选
        pareto_optimal_solutions = self._filter_multi_dimensional_pareto(all_candidate_solutions)
        
        # Step 6: 记忆化存储
        self.pareto_memo[state_key] = pareto_optimal_solutions
        self.solver_stats['pareto_solutions_generated'] += len(pareto_optimal_solutions)
        
        return pareto_optimal_solutions
    
    def _combine_heterogeneous_objectives(self,
                                        immediate_rewards: np.ndarray,
                                        immediate_types: List[ObjectiveType],
                                        future_rewards: np.ndarray, 
                                        future_types: List[ObjectiveType],
                                        transition_prob: float) -> Tuple[np.ndarray, List[ObjectiveType]]:
        """组合不同类型的目标向量"""
        
        # 创建统一的目标空间
        all_objective_types = list(set(immediate_types + future_types))
        combined_vector = np.zeros(len(all_objective_types))
        
        # 映射当前奖励
        for i, obj_type in enumerate(immediate_types):
            global_index = all_objective_types.index(obj_type)
            combined_vector[global_index] += immediate_rewards[i]
        
        # 映射未来奖励
        for i, obj_type in enumerate(future_types):
            if obj_type in all_objective_types:
                global_index = all_objective_types.index(obj_type)
                combined_vector[global_index] += self.gamma * transition_prob * future_rewards[i]
        
        return combined_vector, all_objective_types
    
    def _calculate_tool_specific_score(self,
                                     action: AgentAction,
                                     objectives: np.ndarray,
                                     objective_types: List[ObjectiveType],
                                     state: QuantitativeState) -> float:
        """计算工具特定评分"""
        
        # 获取动态权重
        weights = self.tool_objectives.weight_manager.get_dynamic_weights(action, state)
        
        # 计算加权评分
        weighted_score = 0.0
        for i, obj_type in enumerate(objective_types):
            weight = weights.get(obj_type, 0.25)  # 默认权重
            weighted_score += weight * objectives[i]
        
        return weighted_score
    
    def _filter_multi_dimensional_pareto(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """多维度帕累托筛选"""
        if not solutions:
            return []
        
        # 按工具类型分组
        tool_groups = defaultdict(list)
        for sol in solutions:
            tool_groups[sol.action].append(sol)
        
        pareto_optimal = []
        
        # 每个工具内部先筛选帕累托最优
        for action, tool_solutions in tool_groups.items():
            tool_pareto = self._filter_pareto_within_tool(tool_solutions)
            pareto_optimal.extend(tool_pareto)
        
        # 跨工具的最终帕累托筛选
        return self._filter_cross_tool_pareto(pareto_optimal)
    
    def _filter_pareto_within_tool(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """工具内部帕累托筛选"""
        if not solutions:
            return []
        
        pareto_optimal = []
        
        for i, sol1 in enumerate(solutions):
            is_dominated = False
            
            for j, sol2 in enumerate(solutions):
                if i != j and self._solution_dominates(sol2, sol1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(sol1)
        
        return pareto_optimal
    
    def _filter_cross_tool_pareto(self, solutions: List[ParetoSolution]) -> List[ParetoSolution]:
        """跨工具帕累托筛选"""
        if not solutions:
            return []
        
        # 对于不同工具的解，使用通用目标空间比较
        unified_solutions = []
        
        for sol in solutions:
            # 转换为通用目标空间 [准确性, 效率, 完整性, 成本]
            unified_objectives = self._convert_to_unified_objectives(sol)
            unified_sol = ParetoSolution(
                action=sol.action,
                state=sol.state,
                objectives=unified_objectives,
                objective_types=[ObjectiveType.ACCURACY, ObjectiveType.EFFICIENCY, 
                               ObjectiveType.COMPLETENESS, ObjectiveType.COST],
                policy_path=sol.policy_path,
                expected_reward=sol.expected_reward,
                confidence=sol.confidence,
                tool_specific_score=sol.tool_specific_score
            )
            unified_solutions.append(unified_sol)
        
        # 在统一空间中筛选帕累托最优
        return self._filter_pareto_within_tool(unified_solutions)
    
    def _convert_to_unified_objectives(self, solution: ParetoSolution) -> np.ndarray:
        """将工具特定目标转换为统一目标空间"""
        
        # 创建映射规则
        mapping = {
            ObjectiveType.CLARITY: ObjectiveType.ACCURACY,
            ObjectiveType.COVERAGE: ObjectiveType.COMPLETENESS,
            ObjectiveType.ACTIONABILITY: ObjectiveType.EFFICIENCY,
            ObjectiveType.RELEVANCE: ObjectiveType.ACCURACY,
            ObjectiveType.AUTHORITY: ObjectiveType.ACCURACY,
            ObjectiveType.NOVELTY: ObjectiveType.COMPLETENESS,
            ObjectiveType.RECALL: ObjectiveType.COMPLETENESS,
            ObjectiveType.PRECISION: ObjectiveType.ACCURACY,
            ObjectiveType.FACTUALITY: ObjectiveType.ACCURACY,
            ObjectiveType.COHERENCE: ObjectiveType.ACCURACY,
            ObjectiveType.COMPREHENSIVENESS: ObjectiveType.COMPLETENESS,
            ObjectiveType.FLUENCY: ObjectiveType.EFFICIENCY
        }
        
        # 初始化统一目标向量
        unified = np.zeros(4)  # [准确性, 效率, 完整性, 成本]
        unified_counts = np.zeros(4)
        
        # 映射目标值
        for i, obj_type in enumerate(solution.objective_types):
            if obj_type == ObjectiveType.COST:
                unified[3] += solution.objectives[i]
                unified_counts[3] += 1
            else:
                target_type = mapping.get(obj_type, ObjectiveType.ACCURACY)
                target_index = {
                    ObjectiveType.ACCURACY: 0,
                    ObjectiveType.EFFICIENCY: 1,
                    ObjectiveType.COMPLETENESS: 2,
                    ObjectiveType.COST: 3
                }[target_type]
                
                unified[target_index] += solution.objectives[i]
                unified_counts[target_index] += 1
        
        # 取平均值（避免重复计数）
        for i in range(4):
            if unified_counts[i] > 0:
                unified[i] /= unified_counts[i]
        
        return unified
    
    def _solution_dominates(self, sol1: ParetoSolution, sol2: ParetoSolution) -> bool:
        """判断解1是否支配解2"""
        
        # 如果是相同工具，直接比较目标向量
        if sol1.action == sol2.action:
            return self._objectives_dominate(sol1.objectives, sol2.objectives)
        
        # 如果是不同工具，转换为统一空间比较
        unified1 = self._convert_to_unified_objectives(sol1)
        unified2 = self._convert_to_unified_objectives(sol2)
        
        return self._objectives_dominate(unified1, unified2)
    
    def _objectives_dominate(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """判断目标向量1是否支配目标向量2"""
        all_geq = np.all(obj1 >= obj2)
        any_greater = np.any(obj1 > obj2)
        return all_geq and any_greater
    
    # 其他方法保持不变...
    def _get_feasible_actions(self, state: QuantitativeState) -> List[AgentAction]:
        """获取可行动作集合"""
        all_actions = [
            AgentAction.DECOMPOSE,
            AgentAction.SEARCH_KG, 
            AgentAction.SEARCH_VECTOR,
            AgentAction.GENERATE,
            AgentAction.REFINE
        ]
        
        feasible = []
        for action in all_actions:
            if self.feasibility_checker.is_feasible(state, action):
                feasible.append(action)
        
        return feasible
    
    def _is_terminal_state(self, state: QuantitativeState) -> bool:
        """判断是否为终止状态"""
        return (
            (state.facts_completeness >= 0.8 and state.confidence_level >= 0.7) or
            state.time_elapsed >= 0.95 or
            state.stage >= self.max_stages
        )
    
    def _get_terminal_solutions(self, state: QuantitativeState) -> List[ParetoSolution]:
        """获取终止状态的解"""
        terminal_rewards, terminal_types = self.tool_objectives.calculate_tool_objectives(
            state, AgentAction.GENERATE
        )
        
        return [ParetoSolution(
            action=AgentAction.GENERATE,
            state=state,
            objectives=terminal_rewards,
            objective_types=terminal_types,
            policy_path=[AgentAction.GENERATE],
            expected_reward=np.sum(terminal_rewards),
            confidence=state.confidence_level,
            tool_specific_score=np.mean(terminal_rewards)
        )]

# ===== 6. 状态转移模型 (保持不变) =====
class StateTransitionModel:
    """状态转移模型"""
    
    def get_next_states(self, current_state: QuantitativeState, action: AgentAction) -> List[Tuple[QuantitativeState, float]]:
        """获取下一状态分布"""
        
        if action == AgentAction.DECOMPOSE:
            next_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=current_state.facts_completeness,
                query_complexity=max(0.2, current_state.query_complexity - 0.3),
                search_exhaustion=current_state.search_exhaustion,
                time_elapsed=current_state.time_elapsed + 0.1,
                confidence_level=current_state.confidence_level + 0.1
            )
            return [(next_state, 1.0)]
            
        elif action == AgentAction.SEARCH_KG:
            base_facts_gain = 0.3
            base_time_cost = 0.2
            
            high_gain_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain + 0.2),
                query_complexity=current_state.query_complexity,
                search_exhaustion=min(1.0, current_state.search_exhaustion + 0.3),
                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost),
                confidence_level=min(1.0, current_state.confidence_level + 0.2)
            )
            
            low_gain_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain - 0.1),
                query_complexity=current_state.query_complexity,
                search_exhaustion=min(1.0, current_state.search_exhaustion + 0.2),
                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost + 0.1),
                confidence_level=min(1.0, current_state.confidence_level + 0.05)
            )
            
            return [(high_gain_state, 0.6), (low_gain_state, 0.4)]
            
        elif action == AgentAction.SEARCH_VECTOR:
            base_facts_gain = 0.25
            base_time_cost = 0.15
            
            good_result_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain + 0.1),
                query_complexity=current_state.query_complexity,
                search_exhaustion=min(1.0, current_state.search_exhaustion + 0.25),
                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost),
                confidence_level=min(1.0, current_state.confidence_level + 0.15)
            )
            
            poor_result_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=min(1.0, current_state.facts_completeness + base_facts_gain - 0.05),
                query_complexity=current_state.query_complexity,
                search_exhaustion=min(1.0, current_state.search_exhaustion + 0.2),
                time_elapsed=min(1.0, current_state.time_elapsed + base_time_cost + 0.05),
                confidence_level=min(1.0, current_state.confidence_level + 0.08)
            )
            
            return [(good_result_state, 0.7), (poor_result_state, 0.3)]
        
        elif action == AgentAction.GENERATE:
            terminal_state = QuantitativeState(
                stage=current_state.stage + 1,
                facts_completeness=current_state.facts_completeness,
                query_complexity=current_state.query_complexity,
                search_exhaustion=current_state.search_exhaustion,
                time_elapsed=min(1.0, current_state.time_elapsed + 0.1),
                confidence_level=current_state.confidence_level
            )
            return [(terminal_state, 1.0)]
        
        return [(current_state, 1.0)]

# ===== 7. 可行性检查器 (保持不变) =====
class ActionFeasibilityChecker:
    """动作可行性检查器"""
    
    def __init__(self):
        self.constraints = {
            'max_time': 0.95,
            'min_confidence_for_generation': 0.6,
            'max_search_exhaustion': 0.9,
            'max_stages': 10
        }
    
    def is_feasible(self, state: QuantitativeState, action: AgentAction) -> bool:
        """检查动作是否可行"""
        
        if (state.time_elapsed >= self.constraints['max_time'] or 
            state.stage >= self.constraints['max_stages']):
            return action == AgentAction.GENERATE
        
        if action == AgentAction.DECOMPOSE:
            return state.query_complexity > 0.5 and state.stage <= 1
            
        elif action == AgentAction.SEARCH_KG:
            return state.search_exhaustion < self.constraints['max_search_exhaustion']
            
        elif action == AgentAction.SEARCH_VECTOR:
            return state.search_exhaustion < self.constraints['max_search_exhaustion']
            
        elif action == AgentAction.GENERATE:
            return (state.confidence_level >= self.constraints['min_confidence_for_generation'] or
                   state.time_elapsed >= 0.8)
            
        elif action == AgentAction.REFINE:
            return state.facts_completeness > 0.3
        
        return True

# ===== 8. 主测试函数 =====
def main():
    """测试增强的多目标动态规划系统"""
    
    print("增强的工具特定多目标动态规划系统测试")
    print("="*60)
    
    # 初始化系统
    solver = EnhancedMultiObjectiveDynamicProgramming()
    
    # 测试状态
    test_states = [
        QuantitativeState(
            stage=0,
            facts_completeness=0.1,
            query_complexity=0.8,  # 高复杂度查询
            search_exhaustion=0.0,
            time_elapsed=0.0,
            confidence_level=0.3
        ),
        QuantitativeState(
            stage=2,
            facts_completeness=0.5,
            query_complexity=0.4,  # 中等复杂度
            search_exhaustion=0.3,
            time_elapsed=0.4,
            confidence_level=0.6
        )
    ]
    
    for i, state in enumerate(test_states):
        print(f"\n--- 测试状态 {i+1} ---")
        print(f"查询复杂度: {state.query_complexity:.1f}")
        print(f"事实完整度: {state.facts_completeness:.1f}")
        print(f"时间消耗: {state.time_elapsed:.1f}")
        
        # 求解
        start_time = time.time()
        solutions = solver.solve(state)
        solve_time = time.time() - start_time
        
        print(f"\n求解结果:")
        print(f"- 求解时间: {solve_time:.3f}秒")
        print(f"- 帕累托解数量: {len(solutions)}")
        
        # 显示前3个解的详细信息
        for j, sol in enumerate(solutions[:3]):
            print(f"\n解 {j+1}:")
            print(f"  动作: {sol.action}")
            print(f"  目标类型: {[obj.value for obj in sol.objective_types]}")
            print(f"  目标值: {sol.objectives}")
            print(f"  工具特定评分: {sol.tool_specific_score:.3f}")
            print(f"  策略路径: {sol.policy_path[:3]}")
        
        # 显示目标维度统计
        print(f"\n目标维度使用统计:")
        for obj_type, count in solver.solver_stats['objective_dimensions_used'].items():
            print(f"  {obj_type}: {count}次")
    
    print(f"\n{'='*60}")
    print("测试完成")

if __name__ == "__main__":
    main()