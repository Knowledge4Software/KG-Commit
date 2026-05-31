from typing import Dict, Any

class CommitPayloadPrinter:
    """Handles standardized, clear console formatting for extracted Git commit payloads."""
    
    @staticmethod
    def print_payload(commit_payload: Dict[str, Any]) -> None:
        """Pretty-prints the keys, descriptions, and values of a commit payload dictionary."""
        if not commit_payload:
            print("❌ Cannot print payload: Dictionary is empty or None.")
            return

        print("✅ Success! Raw payload retrieved with customized features.")
        print("=" * 70)
        print(f"{'KEY':<25} | VALUE")
        print("=" * 70)
        
        for key, value in commit_payload.items():
            # Clean preview formatting for the massive diff chunks
            if key == "diff":
                preview = value[:100].replace('\n', ' ') + "..." if value else "Empty Diff"
                print(f"{key:<25} | [Length: {len(value)} chars] -> {preview}")
            
            # Clean preview formatting for multi-line log messages
            elif key == "message":
                first_line = value.strip().split('\n')[0] if value else "Empty Message"
                print(f"{key:<25} | {first_line} ...")
            
            # Formatting collections for easier reading
            elif isinstance(value, list):
                if len(value) > 4:
                    print(f"{key:<25} | [List of {len(value)} items]: {value[:4]} ...")
                else:
                    print(f"{key:<25} | {value}")
                    
            # Fallback for basic data primitives
            else:
                print(f"{key:<25} | {value}")
                
        print("=" * 70)