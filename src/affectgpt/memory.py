from dataclasses import dataclass, field
from typing import List, Dict
@dataclass
class Turn: 
    role: str; text: str; emotions: Dict[str,float]
@dataclass
class Memory:
    short: List[Turn] = field(default_factory=list); max_len: int = 12
    def add(self, role, text, emotions):
        self.short.append(Turn(role, text, emotions))
        if len(self.short)>self.max_len: self.short.pop(0)
    def recent(self): return self.short[-self.max_len:]
