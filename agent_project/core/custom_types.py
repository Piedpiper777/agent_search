"""定义了项目中使用的自定义类型和别名"""
from typing import Dict, List, Any, Union, TypeVar, Callable, TypedDict, Optional, Tuple
from enum import Enum
import numpy as np
from dataclasses import dataclass

# ===== 基础类型别名 =====
JSON = Dict[str, Any]
MetaData = Dict[str, Any]

# ===== 枚举类型 =====
class AgentAction(Enum):
    """Agent可执行的动作类型"""
    # AgentAction: 优化算法决定的"应该做什么"
    DECOMPOSE = "decompose"           # 分解复杂查询
    SEARCH_KG = "search_kg"           # 知识图谱搜索·
    SEARCH_VECTOR = "search_vector"   # 向量搜索
    GENERATE = "generate"             # 生成答案
    REFINE = "refine"                 # 精化结果
    TERMINATE = "terminate"           # 终止执行

class MessageRole(Enum):
    """消息角色类型"""
    USER = "user"                     # 用户消息
    ASSISTANT = "assistant"           # 助手消息 面向用户的最终回复
    CONTROLLER = "controller"         # 控制器消息
    TOOL = "tool"                     # 工具消息
    SYSTEM = "system"                 # 系统消息 系统配置、错误处理、状态管理

class MessageAction(Enum):
    """消息动作类型"""
    # MessageAction: 消息系统中"这条消息做了什么"
    QUERY = "query"                   # 查询
    DECOMPOSE = "decompose"           # 分解
    SEARCH_KG = "search_kg"           # KG搜索
    SEARCH_VECTOR = "search_vector"   # 向量搜索
    GENERATE = "generate"             # 生成
    REFINE = "refine"                 # 精化
    RESPOND = "respond"               # 响应

class OptimizationMethod(Enum):
    """优化方法类型"""
    DYNAMIC_PROGRAMMING = "dp"        # 动态规划
    REINFORCEMENT_LEARNING = "rl"     # 强化学习
    LLM_REASONING = "llm"             # LLM推理
    HYBRID = "hybrid"                 # 混合方法

class ObjectiveType(Enum):
    """目标类型枚举"""
    # ===== 通用目标（所有工具共用的基础目标）=====
    ACCURACY = "accuracy"               # 准确性
    EFFICIENCY = "efficiency"           # 效率 
    COMPLETENESS = "completeness"       # 完整性
    COST = "cost"                      # 成本
    
    # ===== 分解工具特定目标 =====
    CLARITY = "clarity"                # 分解清晰度
    COVERAGE = "coverage"              # 问题覆盖度
    ACTIONABILITY = "actionability"    # 可操作性
    
    # ===== 搜索工具特定目标 =====
    RELEVANCE = "relevance"            # 搜索相关性
    AUTHORITY = "authority"            # 信息权威性
    NOVELTY = "novelty"               # 信息新颖性
    RECALL = "recall"                 # 召回率
    PRECISION = "precision"           # 精确率
    
    # ===== 生成工具特定目标 =====
    COHERENCE = "coherence"           # 生成连贯性
    FLUENCY = "fluency"              # 表达流畅性
    FACTUALITY = "factuality"        # 事实准确性
    COMPREHENSIVENESS = "comprehensiveness"  # 全面性
    
    # ===== 精化工具特定目标 =====
    REFINEMENT_QUALITY = "refinement_quality"  # 精化质量
    CONSISTENCY = "consistency"        # 一致性

# ===== 优化相关类型 =====
ObjectiveVector = np.ndarray
"""统一目标向量类型：[准确性, 效率, 完整性, 成本]"""

"""
用户评价一个AI助手时最关心的四个方面：
- 答案对不对？ → 准确性 (Accuracy)
- 响应快不快？ → 效率 (Efficiency) 
- 信息全不全？ → 完整性 (Completeness)
- 消耗大不大？ → 成本 (Cost)
"""

class ObjectiveScores(TypedDict):
    """目标得分类型（通用四维）"""
    # 用于强化学习和决策分析的统一目标向量ObjectiveVector，例如：[0.8, 0.9, 0.95, -0.1]
    accuracy: float      # 准确性 [0,1]
    efficiency: float    # 效率 [0,1]
    completeness: float  # 完整性 [0,1]
    cost: float          # 成本 [-1,0] (负值表示消耗)

@dataclass
class QuantitativeState:
    """量化状态表示"""
    stage: int                      # 决策阶段 t=0,1,2,...
    facts_completeness: float       # 事实完整度 [0,1]
    query_complexity: float         # 查询复杂度 [0,1]
    search_exhaustion: float        # 搜索穷尽度 [0,1]
    time_elapsed: float             # 时间消耗比例 [0,1]
    confidence_level: float         # 当前置信度 [0,1]
    context: Optional[MetaData] = None     # 上下文信息
    uncertainty_level: float = 0.0        # 不确定性水平 [0,1]
    resource_used: float = 0.0            # 资源使用率 [0,1]
    last_action: Optional[AgentAction] = None  # 上一个执行的动作
    
    def to_key(self) -> str:
        """转换为状态键值用于记忆化"""
        return f"{self.stage}_{self.facts_completeness:.1f}_{self.query_complexity:.1f}_{self.time_elapsed:.1f}_{self.search_exhaustion:.1f}"
    
    def to_vector(self) -> np.ndarray:
        """转换为向量形式（用于RL）"""
        return np.array([
            self.stage / 10.0,  # 归一化阶段数,[0,1]
            self.facts_completeness,
            self.query_complexity,
            self.search_exhaustion,
            self.time_elapsed,
            self.confidence_level,
            self.uncertainty_level,
            self.resource_used
        ])
    
    def clone(self) -> 'QuantitativeState':
        """创建状态副本"""
        return QuantitativeState(
            stage=self.stage,
            facts_completeness=self.facts_completeness,
            query_complexity=self.query_complexity,
            search_exhaustion=self.search_exhaustion,
            time_elapsed=self.time_elapsed,
            confidence_level=self.confidence_level,
            context=self.context.copy() if self.context else None,
            uncertainty_level=self.uncertainty_level,
            resource_used=self.resource_used,
            last_action=self.last_action
        )
    
    def is_terminal(self) -> bool:
        """判断是否为终止状态"""
        return (self.time_elapsed >= 1.0 or 
                self.facts_completeness >= 0.95 or
                self.stage >= 10) # 终止条件：1. 时间消耗达到100%，2. 事实完整度超过95%，3. 达到最大阶段数10

# 工具特定的目标得分类型
class DecomposeObjectives(TypedDict):
    """分解工具目标得分"""
    clarity: float              # 分解清晰度 [0,1]
    coverage: float            # 问题覆盖度 [0,1]
    actionability: float       # 可操作性 [0,1]
    cost: float                # 成本 [-1,0]

class SearchKGObjectives(TypedDict):
    """知识图谱搜索目标得分"""
    relevance: float           # 相关性 [0,1]
    authority: float           # 权威性 [0,1]
    recall: float              # 召回率 [0,1]
    cost: float                # 成本 [-1,0]

class SearchVectorObjectives(TypedDict):
    """向量搜索目标得分"""
    relevance: float           # 相关性 [0,1]
    novelty: float             # 新颖性 [0,1]
    precision: float           # 精确率 [0,1]
    cost: float                # 成本 [-1,0]

class GenerateObjectives(TypedDict):
    """生成工具目标得分"""
    factuality: float          # 事实准确性 [0,1]
    coherence: float           # 连贯性 [0,1]
    comprehensiveness: float   # 全面性 [0,1]
    fluency: float             # 流畅性 [0,1]

class RefineObjectives(TypedDict):
    """精化工具目标得分"""
    accuracy: float            # 准确性 [0,1]
    precision: float           # 精确率 [0,1]
    coherence: float           # 连贯性 [0,1]
    cost: float                # 成本 [-1,0]

# 工具特定目标的联合类型
ToolSpecificObjectives = Union[
    ObjectiveScores,           # 通用目标
    DecomposeObjectives,       # 分解目标
    SearchKGObjectives,        # KG搜索目标
    SearchVectorObjectives,    # 向量搜索目标
    GenerateObjectives,        # 生成目标
    RefineObjectives           # 精化目标
]

@dataclass
class ParetoSolution:
    """帕累托最优解"""
    action: AgentAction
    state: QuantitativeState
    objectives: ObjectiveVector          # [准确性, 效率, 完整性, 成本]
    policy_path: List[AgentAction]       # 从当前状态到终点的动作序列
    expected_reward: float               # 期望总奖励
    confidence: float                    # 解的置信度
    metadata: Optional[MetaData] = None  # 附加信息

class StateTransition(TypedDict):
    """状态转移类型"""
    from_state: QuantitativeState
    action: AgentAction
    to_state: QuantitativeState
    probability: float                   # 转移概率
    reward: ObjectiveVector              # 奖励向量

# ===== 强化学习相关类型 =====
QVector = np.ndarray
"""Q值向量类型：多目标Q值"""

class Experience(TypedDict):
    """强化学习经验类型"""
    state: QuantitativeState
    action: AgentAction
    reward: ObjectiveVector                      # 统一的四维奖励向量
    tool_specific_reward: ToolSpecificObjectives # 工具特定的原始奖励
    objective_types: List[ObjectiveType]         # 使用的目标类型
    next_state: QuantitativeState
    done: bool

class RLPolicy(TypedDict):
    """强化学习策略类型"""
    state_action_values: Dict[str, List[QVector]]  # 状态-动作Q值集合
    weights: ObjectiveVector                       # 目标权重
    exploration_rate: float                        # 探索率

@dataclass
class ParetoQValue:
    """帕累托Q值（增强版）"""
    objectives: ObjectiveVector              # 标准化的四维Q值向量
    tool_specific_objectives: ToolSpecificObjectives  # 工具特定的原始Q值
    objective_types: List[ObjectiveType]     # 目标类型列表
    visit_count: int                        # 访问次数
    last_updated: float                     # 最后更新时间
    confidence: float                       # Q值置信度
    
    def dominates(self, other: 'ParetoQValue') -> bool:
        """判断是否帕累托支配另一个Q值"""
        return (np.all(self.objectives >= other.objectives) and 
                np.any(self.objectives > other.objectives))

class ExplorationStrategy(Enum):
    """探索策略类型"""
    EPSILON_GREEDY = "epsilon_greedy"     # ε-贪婪
    UCB = "ucb"                           # 上置信界
    THOMPSON_SAMPLING = "thompson"        # 汤普森采样
    PARETO_UCB = "pareto_ucb"            # 帕累托UCB

@dataclass
class RLHyperParams:
    """强化学习超参数"""
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    buffer_capacity: int = 10000
    batch_size: int = 32
    update_frequency: int = 100

# ===== 决策相关类型 =====
class DecisionContext(TypedDict):
    """决策上下文类型"""
    query_type: str                           # 查询类型
    time_pressure: float                      # 时间压力 [0,1]
    user_preferences: Dict[str, float]        # 用户偏好，强化学习中使用
    resource_constraints: Dict[str, float]    # 资源约束
    historical_performance: Dict[str, float]  # 历史性能

class DecisionResult(TypedDict):
    """决策结果类型"""
    chosen_action: AgentAction
    reasoning: str                      # 决策理由
    confidence: float                   # 决策置信度
    alternatives: List[ParetoSolution]  # 备选方案
    method_used: OptimizationMethod     # 使用的优化方法
    execution_time: float               # 决策时间(秒)

class DecisionMethod(Enum):
    """决策方法类型"""
    DP_ONLY = "dp_only"                   # 仅DP
    RL_ONLY = "rl_only"                   # 仅RL
    LLM_ONLY = "llm_only"                 # 仅LLM
    DP_LLM = "dp_llm"                     # DP+LLM
    RL_LLM = "rl_llm"                     # RL+LLM
    HYBRID = "hybrid"                     # DP+RL+LLM

@dataclass
class DecisionInput:
    """决策输入数据"""
    state: QuantitativeState
    user_query: str
    context: DecisionContext
    available_actions: List[AgentAction]
    time_budget: float                    # 决策时间预算
    
class LLMDecisionPrompt(TypedDict):
    """LLM决策提示类型"""
    system_message: str                   # 系统消息
    user_context: str                     # 用户上下文
    state_description: str                # 状态描述
    dp_recommendations: str               # DP推荐
    rl_recommendations: str               # RL推荐
    constraint_descriptions: str          # 约束描述
    output_format: str                    # 输出格式要求

# ===== 工具相关类型 (扩展原有) =====
class ToolCategory(Enum):
    """工具类别"""
    SEARCH = "search"                     # 搜索类工具
    ANALYSIS = "analysis"                 # 分析类工具
    GENERATION = "generation"             # 生成类工具
    PROCESSING = "processing"             # 处理类工具

class ToolResult(TypedDict):
    """工具执行结果类型（增强版）"""
    success: bool
    data: Any
    error: Optional[str]
    metadata: MetaData
    execution_time: float                 # 执行时间
    quality_score: float                  # 结果质量评分
    facts_extracted: List[Dict[str, Any]] # 提取的事实
    confidence: float                     # 结果置信度
    coverage: float                       # 覆盖度
    novelty: float                        # 新颖度
    resource_cost: Dict[str, float]       # 实际资源消耗
    state_impact: Dict[str, float]        # 对状态的影响
    
    # 新增目标相关字段
    tool_specific_objectives: ToolSpecificObjectives  # 工具特定目标评分
    unified_objectives: ObjectiveVector                # 统一目标向量

@dataclass
class ToolExecutionContext:
    """工具执行上下文"""
    current_state: QuantitativeState
    user_query: str
    previous_results: List[ToolResult]
    time_remaining: float
    resource_budget: Dict[str, float]     # 资源预算
    quality_requirements: ObjectiveScores # 质量要求
    
class ToolConfig(TypedDict):
    """工具配置类型"""
    name: str
    description: str
    required_fields: List[str]
    optional_fields: List[str]
    expected_execution_time: float      # 预期执行时间
    resource_cost: float                # 资源成本

ToolFunction = Callable[[Any], ToolResult]

# ===== 搜索相关类型 =====
class SearchResult(TypedDict):
    """搜索结果类型"""
    facts: List[Dict[str, Any]]
    confidence: float
    source: str
    metadata: MetaData
    relevance_scores: List[float]       # 相关性分数
    diversity_score: float              # 多样性分数
    coverage_score: float               # 覆盖度分数

class SearchQuery(TypedDict):
    """搜索查询类型"""
    text: str                           # 查询文本
    filters: Dict[str, Any]             # 过滤条件
    top_k: int                          # 返回数量
    search_type: str                    # 搜索类型
    optimization_target: str            # 优化目标 (speed/accuracy/coverage)

# ===== LLM相关类型 =====
class LLMResponse(TypedDict):
    """LLM响应类型"""
    text: str
    confidence: float
    metadata: MetaData
    token_count: int                    # token数量
    response_time: float                # 响应时间
    reasoning_steps: List[str]          # 推理步骤

class LLMRequest(TypedDict):
    """LLM请求类型"""
    messages: List[Dict[str, str]]      # 消息列表
    temperature: float                  # 温度参数
    max_tokens: int                     # 最大token数
    response_format: Optional[Dict[str, str]]  # 响应格式
    optimization_mode: str              # 优化模式 (speed/quality/balanced)

# ===== 监控相关类型 =====
class PerformanceMetrics(TypedDict):
    """性能指标类型"""
    accuracy: float                     # 准确性
    response_time: float                # 响应时间
    resource_usage: float               # 资源使用率
    user_satisfaction: float            # 用户满意度
    cost_efficiency: float              # 成本效率
    
class EpisodeData(TypedDict):
    """Episode数据类型"""
    episode_id: str
    initial_query: str
    actions_taken: List[AgentAction]
    states_sequence: List[QuantitativeState]
    rewards_received: List[ObjectiveVector]
    final_result: str
    performance_metrics: PerformanceMetrics
    total_time: float

@dataclass
class DecisionAnalysis:
    """决策分析结果"""
    decision_quality: float              # 决策质量评分
    consistency_score: float             # 一致性评分（DP/RL/LLM之间）
    robustness_score: float              # 鲁棒性评分
    explanation_quality: float           # 解释质量
    component_contributions: Dict[str, float]  # 各组件贡献度
    
class SystemHealth(TypedDict):
    """系统健康状态"""
    dp_solver_status: str                # DP求解器状态
    rl_learning_progress: float          # RL学习进度
    llm_response_quality: float          # LLM响应质量
    overall_performance: float           # 整体性能
    error_rate: float                    # 错误率
    resource_utilization: Dict[str, float]  # 资源利用率

class AblationResult(TypedDict):
    """消融实验结果"""
    configuration: str                   # 配置名称
    enabled_components: List[str]        # 启用的组件
    performance_metrics: PerformanceMetrics
    decision_accuracy: float             # 决策准确性
    execution_time: float                # 执行时间
    resource_efficiency: float           # 资源效率
    user_satisfaction: float             # 用户满意度

# ===== 状态相关类型 =====
StateData = TypeVar('StateData')

class ValidationResult(TypedDict):
    """验证结果类型"""
    valid: bool
    errors: List[str]
    warnings: List[str]                 # 警告信息
    suggestions: List[str]              # 改进建议

# ===== 配置相关类型 =====
class OptimizationConfig(TypedDict):
    """优化配置类型"""
    dp_max_stages: int                  # DP最大阶段数
    dp_discount_factor: float           # DP折扣因子
    rl_learning_rate: float             # RL学习率
    rl_epsilon: float                   # RL探索率
    objective_weights: ObjectiveScores  # 目标权重
    time_limit: float                   # 时间限制(秒)
    quality_threshold: float            # 质量阈值

class AgentConfig(TypedDict):
    """Agent配置类型"""
    max_iterations: int                       # 最大迭代次数
    termination_conditions: Dict[str, float]  # 终止条件
    optimization_method: OptimizationMethod   # 优化方法
    tools_enabled: List[str]                  # 启用的工具
    monitoring_enabled: bool                  # 是否启用监控

# ===== 回调函数类型 =====
StateUpdateCallback = Callable[[QuantitativeState], None]
"""状态更新回调函数类型"""

DecisionCallback = Callable[[DecisionResult], None]
"""决策回调函数类型"""

PerformanceCallback = Callable[[PerformanceMetrics], None]
"""性能监控回调函数类型"""

# ===== 错误处理类型 =====
class AgentError(TypedDict):
    """Agent错误类型"""
    error_type: str                     # 错误类型
    error_message: str                  # 错误消息
    stack_trace: Optional[str]          # 堆栈跟踪
    recovery_suggestions: List[str]     # 恢复建议
    error_time: float                   # 错误时间

# ===== 工厂函数和工具函数 =====
def create_empty_objectives() -> ObjectiveScores:
    """创建空的目标得分"""
    return {
        'accuracy': 0.0,
        'efficiency': 0.0,
        'completeness': 0.0,
        'cost': 0.0
    }

def objectives_to_vector(objectives: ObjectiveScores) -> ObjectiveVector:
    """将目标得分转换为向量"""
    return np.array([
        objectives['accuracy'],
        objectives['efficiency'],
        objectives['completeness'],
        objectives['cost']
    ])

def vector_to_objectives(vector: ObjectiveVector) -> ObjectiveScores:
    """将向量转换为目标得分"""
    return {
        'accuracy': float(vector[0]),
        'efficiency': float(vector[1]),
        'completeness': float(vector[2]),
        'cost': float(vector[3])
    }

def create_initial_state(query_complexity: float = 0.5) -> QuantitativeState:
    """创建初始状态"""
    return QuantitativeState(
        stage=0,
        facts_completeness=0.0,
        query_complexity=query_complexity,
        search_exhaustion=0.0,
        time_elapsed=0.0,
        confidence_level=0.1,
        uncertainty_level=0.8,
        resource_used=0.0
    )

def combine_objective_vectors(vectors: List[ObjectiveVector], 
                            weights: Optional[ObjectiveVector] = None) -> ObjectiveVector:
    """组合多个目标向量"""
    if not vectors:
        return np.zeros(4)
    
    if weights is None:
        weights = np.ones(4) / 4  # 等权重
    
    # 加权平均
    combined = np.zeros(4)
    for vector in vectors:
        combined += vector * weights
    
    return combined / len(vectors)

def pareto_dominates(vec1: ObjectiveVector, vec2: ObjectiveVector) -> bool:
    """判断vec1是否帕累托支配vec2"""
    return (np.all(vec1 >= vec2) and np.any(vec1 > vec2))

def calculate_hypervolume(solutions: List[ObjectiveVector], 
                         reference_point: Optional[ObjectiveVector] = None) -> float:
    """计算超体积（简化版）"""
    if not solutions:
        return 0.0
    
    if reference_point is None:
        reference_point = np.zeros(4)
    
    # 简化的超体积计算
    volumes = []
    for sol in solutions:
        volume = np.prod(np.maximum(0, sol - reference_point))
        volumes.append(volume)
    
    return sum(volumes)

def normalize_state_vector(state_vector: np.ndarray) -> np.ndarray:
    """归一化状态向量"""
    # 确保所有值都在[0,1]范围内
    return np.clip(state_vector, 0.0, 1.0)

def state_similarity(state1: QuantitativeState, state2: QuantitativeState) -> float:
    """计算两个状态的相似度"""
    vec1 = state1.to_vector()
    vec2 = state2.to_vector()
    
    # 使用余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 1.0 if np.allclose(vec1, vec2) else 0.0
    
    return dot_product / norm_product

# 类型验证函数
def validate_objective_vector(vector: ObjectiveVector) -> ValidationResult:
    """验证目标向量的有效性"""
    errors = []
    warnings = []
    suggestions = []
    
    if len(vector) != 4:
        errors.append(f"目标向量长度应为4，实际为{len(vector)}")
    
    # 检查数值范围
    for i, val in enumerate(vector[:3]):  # 前三个应该是[0,1]
        if not (0 <= val <= 1):
            warnings.append(f"目标{i}的值{val}超出[0,1]范围")
    
    if vector[3] > 0:  # 成本应该≤0
        warnings.append(f"成本值{vector[3]}应该≤0")
    
    if np.any(np.isnan(vector)):
        errors.append("目标向量包含NaN值")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'suggestions': suggestions
    }

def get_tool_objectives(action: AgentAction) -> List[ObjectiveType]:
    """获取工具的特定目标类型"""
    tool_objective_map = {
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
    
    return tool_objective_map.get(action, [
        ObjectiveType.ACCURACY,
        ObjectiveType.EFFICIENCY,
        ObjectiveType.COMPLETENESS,
        ObjectiveType.COST
    ])

def convert_to_unified_objectives(
    tool_specific_obj: ToolSpecificObjectives,
    action: AgentAction
) -> ObjectiveVector:
    """将工具特定目标转换为统一的四维目标向量"""
    
    # 目标映射规则
    mapping_rules = {
        AgentAction.DECOMPOSE: {
            'clarity': 'accuracy',
            'coverage': 'completeness',
            'actionability': 'efficiency',
            'cost': 'cost'
        },
        AgentAction.SEARCH_KG: {
            'relevance': 'accuracy',
            'authority': 'accuracy',  # 权威性也映射到准确性
            'recall': 'completeness',
            'cost': 'cost'
        },
        AgentAction.SEARCH_VECTOR: {
            'relevance': 'accuracy',
            'novelty': 'completeness',
            'precision': 'accuracy',
            'cost': 'cost'
        },
        AgentAction.GENERATE: {
            'factuality': 'accuracy',
            'coherence': 'accuracy',
            'comprehensiveness': 'completeness',
            'fluency': 'efficiency'
        },
        AgentAction.REFINE: {
            'accuracy': 'accuracy',
            'precision': 'accuracy',
            'coherence': 'accuracy',
            'cost': 'cost'
        }
    }
    
    # 初始化统一目标
    unified = {'accuracy': 0.0, 'efficiency': 0.0, 'completeness': 0.0, 'cost': 0.0}
    counts = {'accuracy': 0, 'efficiency': 0, 'completeness': 0, 'cost': 0}
    
    # 获取映射规则
    rules = mapping_rules.get(action, {})
    
    # 转换目标值
    for tool_obj_name, tool_obj_value in tool_specific_obj.items():
        unified_obj_name = rules.get(tool_obj_name, 'accuracy')  # 默认映射到accuracy
        unified[unified_obj_name] += tool_obj_value
        counts[unified_obj_name] += 1
    
    # 计算平均值（避免除零）
    for obj_name in unified:
        if counts[obj_name] > 0:
            unified[obj_name] /= counts[obj_name]
    
    return np.array([
        unified['accuracy'],
        unified['efficiency'],
        unified['completeness'],
        unified['cost']
    ])

def create_tool_specific_objectives(
    action: AgentAction,
    values: List[float]
) -> ToolSpecificObjectives:
    """根据动作类型和数值创建工具特定目标"""
    
    objective_types = get_tool_objectives(action)
    
    if action == AgentAction.DECOMPOSE:
        return DecomposeObjectives(
            clarity=values[0] if len(values) > 0 else 0.0,
            coverage=values[1] if len(values) > 1 else 0.0,
            actionability=values[2] if len(values) > 2 else 0.0,
            cost=values[3] if len(values) > 3 else 0.0
        )
    elif action == AgentAction.SEARCH_KG:
        return SearchKGObjectives(
            relevance=values[0] if len(values) > 0 else 0.0,
            authority=values[1] if len(values) > 1 else 0.0,
            recall=values[2] if len(values) > 2 else 0.0,
            cost=values[3] if len(values) > 3 else 0.0
        )
    elif action == AgentAction.SEARCH_VECTOR:
        return SearchVectorObjectives(
            relevance=values[0] if len(values) > 0 else 0.0,
            novelty=values[1] if len(values) > 1 else 0.0,
            precision=values[2] if len(values) > 2 else 0.0,
            cost=values[3] if len(values) > 3 else 0.0
        )
    elif action == AgentAction.GENERATE:
        return GenerateObjectives(
            factuality=values[0] if len(values) > 0 else 0.0,
            coherence=values[1] if len(values) > 1 else 0.0,
            comprehensiveness=values[2] if len(values) > 2 else 0.0,
            fluency=values[3] if len(values) > 3 else 0.0
        )
    elif action == AgentAction.REFINE:
        return RefineObjectives(
            accuracy=values[0] if len(values) > 0 else 0.0,
            precision=values[1] if len(values) > 1 else 0.0,
            coherence=values[2] if len(values) > 2 else 0.0,
            cost=values[3] if len(values) > 3 else 0.0
        )
    else:
        # 默认返回通用目标
        return ObjectiveScores(
            accuracy=values[0] if len(values) > 0 else 0.0,
            efficiency=values[1] if len(values) > 1 else 0.0,
            completeness=values[2] if len(values) > 2 else 0.0,
            cost=values[3] if len(values) > 3 else 0.0
        )

def validate_tool_objectives(
    action: AgentAction,
    objectives: ToolSpecificObjectives
) -> ValidationResult:
    """验证工具特定目标的有效性"""
    errors = []
    warnings = []
    suggestions = []
    
    expected_objectives = get_tool_objectives(action)
    
    # 检查目标类型匹配
    if action == AgentAction.DECOMPOSE and not isinstance(objectives, dict):
        errors.append("分解工具需要DecomposeObjectives类型")
    elif action == AgentAction.SEARCH_KG and not isinstance(objectives, dict):
        errors.append("KG搜索工具需要SearchKGObjectives类型")
    # ... 其他工具的检查
    
    # 检查数值范围
    for obj_name, obj_value in objectives.items():
        if obj_name != 'cost' and not (0 <= obj_value <= 1):
            warnings.append(f"{obj_name}的值{obj_value}应在[0,1]范围内")
        elif obj_name == 'cost' and obj_value > 0:
            warnings.append(f"成本值{obj_value}应该≤0")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'suggestions': suggestions
    }