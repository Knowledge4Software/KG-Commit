import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Pattern, Optional


class BaseCommitParser(ABC):
    """Abstract Base Class for all commit parsers to ensure interface consistency."""

    @abstractmethod
    def parse(self, commit_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parses a commit payload and returns extracted entities and metadata."""
        pass

import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Pattern, Optional


class BaseCommitParser(ABC):
    """Abstract Base Class for all commit parsers to ensure interface consistency."""

    @abstractmethod
    def parse(self, commit_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parses a commit payload and returns extracted entities and metadata."""
        pass


class RegexCommitParser(BaseCommitParser):
    """
    An efficient parser that evaluates built-in structural and 
    linguistic regex rules simultaneously across diff streams in O(n) time.
    """

    def __init__(self):
        self.rules: Dict[str, str] = {}
        self._combined_regex: Optional[Pattern] = None
        
        # Load your specific tracking rules automatically
        self._load_built_in_rules()

    def _load_built_in_rules(self) -> None:
        """Defines and registers all baseline multi-line pattern rules."""
        # Kept deliberately empty for now
        pass

    def _compile_rules(self) -> Pattern:
        """Combines all registered built-in rules into a single state machine pattern."""
        if not self.rules:
            # Fallback if no rules are registered yet (matches nothing safely)
            return re.compile(r'(?!b)b') 
            
        # Join patterns using the alternative operator | and give them named groups
        combined_pattern = "|".join(f"(?P<{name}>{pattern})" for name, pattern in self.rules.items())
        return re.compile(combined_pattern, re.MULTILINE)

    def parse(self, commit_payload: Dict[str, Any]) -> Dict[str, Any]:
        diff = commit_payload.get("diff", "")
        
        # Capture ALL key/value pairs dynamically from the raw commit payload dictionary
        extracted_entities: Dict[str, Any] = {
            key: value for key, value in commit_payload.items()
        }

        # Lazily compile the simultaneous regex matrix on first call
        if self._combined_regex is None:
            self._combined_regex = self._compile_rules()

        # Single-pass execution block: O(n) sequence matching over diff text
        for match in self._combined_regex.finditer(diff):
            for group_name, value in match.groupdict().items():
                if value is not None:
                    if group_name not in extracted_entities:
                        extracted_entities[group_name] = []
                    
                    if value not in extracted_entities[group_name]:
                        extracted_entities[group_name].append(value)

        return extracted_entities