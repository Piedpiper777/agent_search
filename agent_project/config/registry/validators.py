from typing import Dict, Callable, Any, Optional
import re
from dataclasses import dataclass

@dataclass
class Validator:
    """验证器定义"""
    func: Callable[[Any], bool]
    description: str
    error_message: str

class ValidatorRegistry:
    """验证器注册表，管理所有输入验证器"""
    
    _validators: Dict[str, Validator] = {
        # 查询验证器
        "query": Validator(
            func=lambda x: isinstance(x, str) and len(x.strip()) > 0,
            description="验证查询字符串",
            error_message="查询不能为空"
        ),
        
        # URL验证器
        "url": Validator(
            func=lambda x: bool(re.match(r'^https?://\S+$', str(x))),
            description="验证URL格式",
            error_message="无效的URL格式"
        ),
        
        # 数值范围验证器
        "number_range": Validator(
            func=lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
            description="验证数值范围(0-1)",
            error_message="数值必须在0到1之间"
        ),
        
        # JSON验证器
        "json": Validator(
            func=lambda x: isinstance(x, (dict, list)),
            description="验证JSON格式",
            error_message="无效的JSON格式"
        )
    }
    
    @classmethod
    def register(
        cls,
        name: str,
        func: Callable[[Any], bool],
        description: str,
        error_message: str
    ) -> None:
        """注册新的验证器"""
        if name in cls._validators:
            raise ValueError(f"验证器 {name} 已存在")
        cls._validators[name] = Validator(func, description, error_message)
    
    @classmethod
    def get_validator(cls, name: str) -> Optional[Validator]:
        """获取验证器"""
        return cls._validators.get(name)
    
    @classmethod
    def validate(cls, name: str, value: Any) -> bool:
        """执行验证"""
        validator = cls.get_validator(name)
        if not validator:
            raise ValueError(f"未知的验证器: {name}")
        return validator.func(value)
    
    @classmethod
    def get_error_message(cls, name: str) -> str:
        """获取验证器的错误信息"""
        validator = cls.get_validator(name)
        if not validator:
            raise ValueError(f"未知的验证器: {name}")
        return validator.error_message

# 使用示例
if __name__ == "__main__":
    # 注册自定义验证器
    ValidatorRegistry.register(
        name="length_limit",
        func=lambda x: isinstance(x, str) and len(x) <= 1000,
        description="验证字符串长度限制",
        error_message="字符串长度超过限制(1000)"
    )
    
    # 测试验证
    test_cases = [
        ("query", ""),                    # 应该失败
        ("query", "有效的查询"),           # 应该成功
        ("url", "https://example.com"),   # 应该成功
        ("url", "invalid-url"),           # 应该失败
        ("number_range", 0.5),            # 应该成功
        ("number_range", 1.5),            # 应该失败
    ]
    
    print("\n验证测试:")
    for validator_name, value in test_cases:
        result = ValidatorRegistry.validate(validator_name, value)
        print(f"- {validator_name}({value}): {'通过' if result else '失败'}")
        if not result:
            print(f"  错误: {ValidatorRegistry.get_error_message(validator_name)}")