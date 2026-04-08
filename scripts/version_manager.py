"""
Model Version Manager

Handles semantic versioning for models:
- Auto-increments version numbers (major.minor.patch)
- Tracks version history with timestamps and metadata
- Compares two versions to identify changes
- Maintains VERSION_HISTORY.md for audit trail

MLOps Best Practice: Every model deployment should have a unique, immutable version
"""

import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys


class ModelVersionManager:
    """Manages model semantic versioning and history"""
    
    CONFIG_PATH = "configs/experiment.yaml"
    VERSION_HISTORY_FILE = "VERSION_HISTORY.md"
    
    def __init__(self):
        self.config_path = Path(self.CONFIG_PATH)
        self.history_path = Path(self.VERSION_HISTORY_FILE)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load experiment configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_config(self) -> None:
        """Save updated configuration"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def get_current_version(self) -> str:
        """Get current model version from config"""
        return self.config['experiment']['version']
    
    def get_git_info(self) -> Dict[str, str]:
        """Get current git commit and branch"""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"]
            ).decode().strip()
            return {"commit": commit, "branch": branch}
        except Exception as e:
            print(f"Warning: Could not get git info: {e}")
            return {"commit": "unknown", "branch": "unknown"}
    
    def increment_version(self, version_type: str = "patch") -> str:
        """
        Increment version number (semantic versioning)
        
        Args:
            version_type: 'major', 'minor', or 'patch'
        
        Returns:
            New version string (e.g., "1.0.1")
        """
        current = self.get_current_version()
        parts = list(map(int, current.split('.')))
        
        if version_type == "major":
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif version_type == "minor":
            parts[1] += 1
            parts[2] = 0
        elif version_type == "patch":
            parts[2] += 1
        else:
            raise ValueError("version_type must be 'major', 'minor', or 'patch'")
        
        new_version = '.'.join(map(str, parts))
        return new_version
    
    def update_version(self, version_type: str = "patch", reason: str = "") -> str:
        """
        Update model version and add to history
        
        Args:
            version_type: 'major', 'minor', or 'patch'
            reason: Description of what changed (e.g., "Improved hyperparameters")
        
        Returns:
            New version string
        """
        new_version = self.increment_version(version_type)
        old_version = self.get_current_version()
        
        # Update config
        self.config['experiment']['version'] = new_version
        self.config['experiment']['created_date'] = datetime.now().strftime("%Y-%m-%d")
        self._save_config()
        
        # Add to history
        self._add_to_history(old_version, new_version, reason)
        
        print(f"✓ Version incremented: {old_version} → {new_version}")
        return new_version
    
    def _add_to_history(self, old_version: str, new_version: str, reason: str = "") -> None:
        """Add version entry to VERSION_HISTORY.md"""
        git_info = self.get_git_info()
        timestamp = datetime.now().isoformat()
        
        entry = f"""
### {new_version} → {old_version}
**Date:** {timestamp}  
**Commit:** `{git_info['commit'][:8]}`  
**Branch:** `{git_info['branch']}`  
**Change Type:** {old_version} → {new_version}  
**Reason:** {reason if reason else 'Regular training run'}
"""
        
        # Prepend to history file
        if self.history_path.exists():
            existing = self.history_path.read_text()
            self.history_path.write_text(entry.strip() + "\n\n" + existing)
        else:
            header = """# Model Version History

Tracks all model versions with training details, performance metrics, and hyperparameters.

---

"""
            self.history_path.write_text(header + entry.strip() + "\n")
    
    def show_version_history(self, limit: int = 10) -> None:
        """Display version history"""
        if not self.history_path.exists():
            print("No version history found. Train a model first.")
            return
        
        content = self.history_path.read_text()
        lines = content.split('\n')[:limit * 5]  # Rough estimate
        print('\n'.join(lines))
    
    def compare_versions(self, version1: str, version2: str) -> None:
        """Compare two versions from git history"""
        # This would require storing version tags in git
        print(f"Comparing {version1} vs {version2}")
        print("(Implementation requires git tags per version)")


def suggest_version_bump(reason: str) -> str:
    """
    Suggest version bump type based on reason
    
    Args:
        reason: Description of changes
    
    Returns:
        Suggested bump type: 'major', 'minor', or 'patch'
    """
    reason_lower = reason.lower()
    
    # Major: Breaking changes, architecture changes
    if any(word in reason_lower for word in ['breaking', 'architecture', 'retraining', 'retrain']):
        return "major"
    
    # Minor: New features, hyperparameter changes
    if any(word in reason_lower for word in ['hyperparameter', 'feature', 'improvement', 'optimize']):
        return "minor"
    
    # Patch: Bug fixes, data fixes
    if any(word in reason_lower for word in ['fix', 'bug', 'patch', 'hotfix', 'data']):
        return "patch"
    
    # Default to patch
    return "patch"


def create_release_tag(version: str, description: str = "") -> None:
    """Create git tag for version release"""
    try:
        tag_msg = f"Release model v{version}\n\n{description}"
        subprocess.run(
            ["git", "tag", "-a", f"model-v{version}", "-m", tag_msg],
            check=True
        )
        print(f"✓ Git tag created: model-v{version}")
    except Exception as e:
        print(f"Warning: Could not create git tag: {e}")


if __name__ == "__main__":
    manager = ModelVersionManager()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python version_manager.py current              - Show current version")
        print("  python version_manager.py bump [type] [reason] - Bump version")
        print("  python version_manager.py history              - Show version history")
        print("\nExample:")
        print("  python version_manager.py bump minor 'Improved hyperparameters'")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "current":
        print(f"Current model version: {manager.get_current_version()}")
    
    elif command == "bump":
        version_type = sys.argv[2] if len(sys.argv) > 2 else "patch"
        reason = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
        new_version = manager.update_version(version_type, reason)
        
        # Create git tag
        create_release_tag(new_version, reason)
    
    elif command == "history":
        manager.show_version_history()
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
