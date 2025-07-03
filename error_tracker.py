#!/usr/bin/env python3
"""
Automated Error Tracking and TODO Management System
Automatically captures errors, adds them to TODO lists, and tracks fixes.
"""

import json
import hashlib
import traceback
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import functools

class ErrorTracker:
    """Automatic error tracking and TODO management system."""
    
    def __init__(self, 
                 errors_file: str = "errors_log.json",
                 todo_file: str = "TODO.md",
                 fixed_file: str = "FIXED_ERRORS.md"):
        self.errors_file = Path(errors_file)
        self.todo_file = Path(todo_file)
        self.fixed_file = Path(fixed_file)
        
        # Initialize files if they don't exist
        self._initialize_files()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_files(self):
        """Initialize tracking files if they don't exist."""
        if not self.errors_file.exists():
            self.errors_file.write_text(json.dumps({
                "active_errors": {},
                "fixed_errors": {},
                "error_count": 0,
                "last_updated": self._get_timestamp()
            }, indent=2))
        
        if not self.fixed_file.exists():
            self.fixed_file.write_text(
                "# Fixed Errors Log\n\n"
                "**Automatically generated log of resolved issues**\n\n"
                "---\n\n"
            )
    
    def _get_timestamp(self) -> str:
        """Get current UTC timestamp."""
        return datetime.datetime.utcnow().isoformat() + "Z"
    
    def _get_error_hash(self, error_type: str, error_msg: str, location: str) -> str:
        """Generate unique hash for error identification."""
        error_string = f"{error_type}:{error_msg}:{location}"
        return hashlib.md5(error_string.encode()).hexdigest()[:8]
    
    def _load_errors(self) -> Dict[str, Any]:
        """Load current errors from JSON file."""
        try:
            return json.loads(self.errors_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {
                "active_errors": {},
                "fixed_errors": {},
                "error_count": 0,
                "last_updated": self._get_timestamp()
            }
    
    def _save_errors(self, data: Dict[str, Any]):
        """Save errors to JSON file."""
        data["last_updated"] = self._get_timestamp()
        self.errors_file.write_text(json.dumps(data, indent=2))
    
    def track_error(self, 
                   error: Exception, 
                   context: str = "",
                   severity: str = "HIGH",
                   auto_add_todo: bool = True) -> str:
        """
        Track a new error and optionally add to TODO list.
        
        Args:
            error: The exception that occurred
            context: Additional context about where/when the error happened
            severity: Error severity (HIGH, MEDIUM, LOW)
            auto_add_todo: Whether to automatically add to TODO list
            
        Returns:
            Error hash ID for tracking
        """
        error_type = type(error).__name__
        error_msg = str(error)
        location = context or self._get_caller_info()
        
        error_hash = self._get_error_hash(error_type, error_msg, location)
        
        # Load current errors
        data = self._load_errors()
        
        # Check if this error already exists
        if error_hash in data["active_errors"]:
            # Update occurrence count
            data["active_errors"][error_hash]["occurrences"] += 1
            data["active_errors"][error_hash]["last_seen"] = self._get_timestamp()
            self.logger.info(f"Updated existing error {error_hash} (occurrence #{data['active_errors'][error_hash]['occurrences']})")
        else:
            # Add new error
            data["error_count"] += 1
            data["active_errors"][error_hash] = {
                "id": error_hash,
                "error_type": error_type,
                "error_message": error_msg,
                "location": location,
                "severity": severity,
                "first_seen": self._get_timestamp(),
                "last_seen": self._get_timestamp(),
                "occurrences": 1,
                "traceback": traceback.format_exc(),
                "status": "ACTIVE",
                "added_to_todo": False
            }
            self.logger.info(f"Tracked new error {error_hash}: {error_type}")
        
        # Save errors
        self._save_errors(data)
        
        # Automatically add to TODO if requested
        if auto_add_todo and not data["active_errors"][error_hash]["added_to_todo"]:
            self._add_error_to_todo(error_hash, data["active_errors"][error_hash])
            data["active_errors"][error_hash]["added_to_todo"] = True
            self._save_errors(data)
        
        return error_hash
    
    def mark_error_fixed(self, error_hash: str, fix_description: str = ""):
        """Mark an error as fixed and move it to fixed errors log."""
        data = self._load_errors()
        
        if error_hash not in data["active_errors"]:
            self.logger.warning(f"Error {error_hash} not found in active errors")
            return False
        
        error_info = data["active_errors"][error_hash]
        error_info["status"] = "FIXED"
        error_info["fixed_date"] = self._get_timestamp()
        error_info["fix_description"] = fix_description
        
        # Move to fixed errors
        data["fixed_errors"][error_hash] = error_info
        del data["active_errors"][error_hash]
        
        # Save updated data
        self._save_errors(data)
        
        # Remove from TODO and add to fixed log
        self._remove_error_from_todo(error_hash)
        self._add_to_fixed_log(error_info)
        
        self.logger.info(f"Marked error {error_hash} as fixed")
        return True
    
    def _add_error_to_todo(self, error_hash: str, error_info: Dict[str, Any]):
        """Add error to TODO.md file."""
        if not self.todo_file.exists():
            self.logger.warning("TODO.md not found, cannot add error")
            return
        
        content = self.todo_file.read_text()
        
        # Create error entry
        error_entry = (
            f"- [ ] **Fix Error {error_hash}** [{error_info['severity']}]\n"
            f"  - **Type**: {error_info['error_type']}\n"
            f"  - **Message**: {error_info['error_message']}\n"
            f"  - **Location**: {error_info['location']}\n"
            f"  - **First Seen**: {error_info['first_seen'][:10]}\n"
            f"  - **Occurrences**: {error_info['occurrences']}\n"
            f"  - **Auto-tracked error** - Fix and run `python3 error_tracker.py fix {error_hash}`\n\n"
        )
        
        # Find the right section to add to based on severity
        if error_info['severity'] == 'HIGH':
            section_marker = "### HIGH PRIORITY"
        elif error_info['severity'] == 'MEDIUM':
            section_marker = "### MEDIUM PRIORITY"
        else:
            section_marker = "### LOW PRIORITY"
        
        if section_marker in content:
            # Insert after the section header
            lines = content.split('\n')
            insert_index = -1
            for i, line in enumerate(lines):
                if line.strip() == section_marker:
                    insert_index = i + 1
                    break
            
            if insert_index > 0:
                lines.insert(insert_index, error_entry.rstrip())
                self.todo_file.write_text('\n'.join(lines))
                self.logger.info(f"Added error {error_hash} to TODO.md under {section_marker}")
            else:
                self.logger.warning(f"Could not find {section_marker} in TODO.md")
        else:
            # Append to end of file
            with open(self.todo_file, 'a') as f:
                f.write(f"\n## AUTO-TRACKED ERRORS\n\n{error_entry}")
            self.logger.info(f"Added error {error_hash} to end of TODO.md")
    
    def _remove_error_from_todo(self, error_hash: str):
        """Remove error from TODO.md file."""
        if not self.todo_file.exists():
            return
        
        content = self.todo_file.read_text()
        lines = content.split('\n')
        
        # Find and remove the error entry
        new_lines = []
        skip_lines = 0
        
        for line in lines:
            if skip_lines > 0:
                skip_lines -= 1
                continue
                
            if f"Fix Error {error_hash}" in line:
                # Skip this line and the next 7 lines (error details)
                skip_lines = 7
                continue
            
            new_lines.append(line)
        
        self.todo_file.write_text('\n'.join(new_lines))
        self.logger.info(f"Removed error {error_hash} from TODO.md")
    
    def _add_to_fixed_log(self, error_info: Dict[str, Any]):
        """Add fixed error to the fixed errors log."""
        fixed_entry = (
            f"## Fixed Error {error_info['id']} ✅\n\n"
            f"**Fixed on**: {error_info['fixed_date'][:10]}\n"
            f"**Type**: {error_info['error_type']}\n"
            f"**Message**: {error_info['error_message']}\n"
            f"**Location**: {error_info['location']}\n"
            f"**First Seen**: {error_info['first_seen'][:10]}\n"
            f"**Total Occurrences**: {error_info['occurrences']}\n"
            f"**Fix Description**: {error_info.get('fix_description', 'No description provided')}\n\n"
            f"---\n\n"
        )
        
        # Prepend to fixed errors file (most recent first)
        if self.fixed_file.exists():
            content = self.fixed_file.read_text()
            # Insert after the initial header
            header_end = content.find("---\n\n") + 5
            new_content = content[:header_end] + fixed_entry + content[header_end:]
            self.fixed_file.write_text(new_content)
        else:
            self.fixed_file.write_text(f"# Fixed Errors Log\n\n---\n\n{fixed_entry}")
        
        self.logger.info(f"Added error {error_info['id']} to fixed errors log")
    
    def _get_caller_info(self) -> str:
        """Get information about where the error tracking was called from."""
        import inspect
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller
            for _ in range(3):  # Skip this method, track_error, and the decorator
                frame = frame.f_back
                if frame is None:
                    break
            
            if frame:
                filename = Path(frame.f_code.co_filename).name
                function_name = frame.f_code.co_name
                line_number = frame.f_lineno
                return f"{filename}:{function_name}:{line_number}"
            else:
                return "unknown_location"
        finally:
            del frame
    
    def get_active_errors(self) -> Dict[str, Any]:
        """Get all currently active errors."""
        data = self._load_errors()
        return data["active_errors"]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of error tracking status."""
        data = self._load_errors()
        
        return {
            "active_errors_count": len(data["active_errors"]),
            "fixed_errors_count": len(data["fixed_errors"]),
            "total_errors_tracked": data["error_count"],
            "last_updated": data["last_updated"],
            "active_errors": data["active_errors"]
        }

# Global tracker instance
_error_tracker = ErrorTracker()

def track_errors(context: str = "", severity: str = "HIGH", auto_add_todo: bool = True):
    """
    Decorator to automatically track errors in functions.
    
    Usage:
        @track_errors(context="image_processing", severity="HIGH")
        def process_image():
            # Your code here
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or f"{func.__module__}.{func.__name__}"
                error_hash = _error_tracker.track_error(e, error_context, severity, auto_add_todo)
                print(f"⚠️  Error {error_hash} tracked and added to TODO list")
                raise  # Re-raise the exception
        return wrapper
    return decorator

@contextmanager
def error_tracking_context(context: str = "", severity: str = "HIGH", auto_add_todo: bool = True):
    """
    Context manager for tracking errors in code blocks.
    
    Usage:
        with error_tracking_context("flux_generation", "HIGH"):
            # Your code here
            pass
    """
    try:
        yield
    except Exception as e:
        error_hash = _error_tracker.track_error(e, context, severity, auto_add_todo)
        print(f"⚠️  Error {error_hash} tracked and added to TODO list")
        raise

def main():
    """CLI interface for error tracker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Error Tracker CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show error tracking status')
    
    # Fix command
    fix_parser = subparsers.add_parser('fix', help='Mark an error as fixed')
    fix_parser.add_argument('error_hash', help='Error hash ID to mark as fixed')
    fix_parser.add_argument('--description', '-d', default='', help='Description of the fix')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List active errors')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        summary = _error_tracker.get_error_summary()
        print("📊 Error Tracking Status")
        print("=" * 30)
        print(f"Active Errors: {summary['active_errors_count']}")
        print(f"Fixed Errors: {summary['fixed_errors_count']}")
        print(f"Total Tracked: {summary['total_errors_tracked']}")
        print(f"Last Updated: {summary['last_updated'][:19]}")
        
    elif args.command == 'fix':
        success = _error_tracker.mark_error_fixed(args.error_hash, args.description)
        if success:
            print(f"✅ Marked error {args.error_hash} as fixed")
        else:
            print(f"❌ Could not find error {args.error_hash}")
            
    elif args.command == 'list':
        errors = _error_tracker.get_active_errors()
        if not errors:
            print("🎉 No active errors!")
        else:
            print("📋 Active Errors:")
            print("=" * 20)
            for error_hash, error_info in errors.items():
                print(f"🔴 {error_hash}: {error_info['error_type']}")
                print(f"   Message: {error_info['error_message'][:60]}...")
                print(f"   Severity: {error_info['severity']}")
                print(f"   Occurrences: {error_info['occurrences']}")
                print()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
