"""
Model Version Diff Viewer

Compare two model versions to see what changed:
- Performance metrics
- Hyperparameters
- Data version
- Feature changes
- Training details
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import sys


class VersionDiffViewer:
    """Compare two model versions"""
    
    METADATA_PATH = "model/metadata.json"
    
    def __init__(self):
        self.current_metadata = self._load_metadata("HEAD")
    
    def _load_metadata(self, commit: str = "HEAD") -> Optional[Dict]:
        """Load metadata.json from a specific git commit"""
        try:
            metadata_json = subprocess.check_output(
                ["git", "show", f"{commit}:{self.METADATA_PATH}"],
                stderr=subprocess.DEVNULL
            ).decode()
            return json.loads(metadata_json)
        except Exception:
            return None
    
    def compare_versions(self, commit1: str, commit2: str = "HEAD") -> None:
        """Compare two model versions from git history"""
        meta1 = self._load_metadata(commit1)
        meta2 = self._load_metadata(commit2)
        
        if not meta1 or not meta2:
            print("❌ Could not load metadata from one or both commits")
            return
        
        print("\n" + "=" * 80)
        print(f"MODEL VERSION COMPARISON")
        print("=" * 80)
        
        # Version comparison
        self._compare_section("VERSION", meta1, meta2, "model_registry")
        
        # Performance comparison
        self._compare_section("PERFORMANCE", meta1, meta2, "model_performance")
        
        # Hyperparameter comparison
        self._compare_section("HYPERPARAMETERS", meta1, meta2, "hyperparameters")
        
        # Data comparison
        self._compare_section("DATA LINEAGE", meta1, meta2, "model_lineage")
        
        # Training details comparison
        self._compare_section("TRAINING INFO", meta1, meta2, "training_info")
    
    def _compare_section(self, section_name: str, meta1: Dict, meta2: Dict, 
                        section_key: str) -> None:
        """Compare a section of metadata"""
        if section_key not in meta1 or section_key not in meta2:
            return
        
        section1 = meta1[section_key]
        section2 = meta2[section_key]
        
        print(f"\n{'='*80}")
        print(f"  {section_name}")
        print(f"{'='*80}")
        
        all_keys = set(section1.keys()) | set(section2.keys())
        
        for key in sorted(all_keys):
            val1 = section1.get(key)
            val2 = section2.get(key)
            
            # Skip nested dicts and lists for now
            if isinstance(val1, (dict, list)) or isinstance(val2, (dict, list)):
                continue
            
            if val1 != val2:
                print(f"  {key:30s}: {str(val1):20s} → {str(val2)}")
            else:
                print(f"  {key:30s}: {str(val1)}")
    
    def show_version_timeline(self) -> None:
        """Show timeline of versions from git history"""
        try:
            # Get all commits that modified metadata.json
            log_output = subprocess.check_output(
                ["git", "log", "--oneline", "--", self.METADATA_PATH]
            ).decode().strip()
            
            if not log_output:
                print("No version history found")
                return
            
            print("\n" + "=" * 80)
            print("  VERSION TIMELINE (from git history)")
            print("=" * 80 + "\n")
            
            commits = log_output.split('\n')[:20]  # Show last 20
            
            for i, line in enumerate(commits, 1):
                parts = line.split(' ', 1)
                commit = parts[0]
                message = parts[1] if len(parts) > 1 else "Training run"
                
                metadata = self._load_metadata(commit)
                if metadata:
                    version = metadata['model_registry'].get('version', 'unknown')
                    roc_auc = metadata['model_performance']['test_set'].get('roc_auc', 'N/A')
                    date = metadata['model_registry'].get('created_date', 'unknown')
                    
                    print(f"{i:2d}. {version:8s} | ROC AUC: {str(roc_auc):6s} | "
                          f"{date} | {commit[:8]} | {message[:40]}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    def show_deployment_history(self) -> None:
        """Show Docker image tags and deployments"""
        try:
            # Get git tags related to models
            tags_output = subprocess.check_output(
                ["git", "tag", "-l", "model-v*"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            if not tags_output:
                print("No deployment tags found")
                return
            
            print("\n" + "=" * 80)
            print("  DEPLOYMENT HISTORY (Git Tags)")
            print("=" * 80 + "\n")
            
            tags = sorted(tags_output.split('\n'), reverse=True)
            
            for tag in tags[:20]:  # Show last 20
                try:
                    # Get tag message and date
                    tag_info = subprocess.check_output(
                        ["git", "tag", "-n1", tag]
                    ).decode().strip()
                    
                    # Try to get commit date
                    commit_date = subprocess.check_output(
                        ["git", "log", "-1", "--format=%ci", tag]
                    ).decode().strip()
                    
                    print(f"  {tag:20s} | {commit_date[:10]} | {tag_info[len(tag):].strip()[:50]}")
                except:
                    pass
        
        except Exception as e:
            print(f"Error: {e}")


def main():
    """CLI interface"""
    viewer = VersionDiffViewer()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python version_diff.py compare <commit1> [commit2]  - Compare versions")
        print("  python version_diff.py timeline                     - Show version timeline")
        print("  python version_diff.py deployments                  - Show deployment history")
        print("\nExample:")
        print("  python version_diff.py compare HEAD~1 HEAD")
        print("  python version_diff.py timeline")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "compare":
        commit1 = sys.argv[2] if len(sys.argv) > 2 else "HEAD~1"
        commit2 = sys.argv[3] if len(sys.argv) > 3 else "HEAD"
        viewer.compare_versions(commit1, commit2)
    
    elif command == "timeline":
        viewer.show_version_timeline()
    
    elif command == "deployments":
        viewer.show_deployment_history()
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
