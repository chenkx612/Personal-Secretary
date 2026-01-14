"""
用户画像管理
"""

import json
import os
from typing import Dict, Any

class UserProfile:
    """用户画像管理"""
    
    def __init__(self, user_name: str, profile_file: str = None):
        """
        初始化用户画像
        
        Args:
            user_name: 用户名
            profile_file: 画像文件路径，默认使用user_profile_<user_name>.json
        """
        self.user_name = user_name
        self.profile_file = profile_file or f"user_profile_{user_name}.json"
        self.profile = self._load_profile()
    
    def _load_profile(self) -> Dict[str, Any]:
        """
        加载用户画像
        
        Returns:
            用户画像字典
        """
        if os.path.exists(self.profile_file):
            with open(self.profile_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 默认画像结构
        return {
            "personal_info": {},
            "interests": [],
            "preferences": {"likes": [], "dislikes": []},
            "goals": [],
            "experiences": [],
            "relationships": [],
            "habits": [],
            "concerns": []
        }
    
    def _save_profile(self) -> None:
        """
        保存用户画像
        """
        with open(self.profile_file, 'w', encoding='utf-8') as f:
            json.dump(self.profile, f, ensure_ascii=False, indent=2)
    
    def update(self, extracted_info: Dict[str, Any]) -> None:
        """
        更新用户画像
        
        Args:
            extracted_info: 提取的用户信息
        """
        for category, value in extracted_info.items():
            if category in self.profile:
                if isinstance(self.profile[category], dict):
                    # 更新字典类型的画像项
                    self.profile[category].update(value)
                elif isinstance(self.profile[category], list):
                    # 更新列表类型的画像项
                    if isinstance(value, list):
                        for item in value:
                            if item not in self.profile[category]:
                                self.profile[category].append(item)
                    else:
                        if value not in self.profile[category]:
                            self.profile[category].append(value)
        
        self._save_profile()
    
    def get_profile(self) -> Dict[str, Any]:
        """
        获取用户画像
        
        Returns:
            用户画像字典
        """
        return self.profile
    
    def clear(self) -> None:
        """
        清除用户画像
        """
        self.profile = self._load_profile()  # 重置为默认值
        if os.path.exists(self.profile_file):
            os.remove(self.profile_file)
    
    def to_string(self) -> str:
        """
        将用户画像转换为字符串格式
        
        Returns:
            用户画像字符串
        """
        return json.dumps(self.profile, ensure_ascii=False, indent=2)