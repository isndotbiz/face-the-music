#!/usr/bin/env python3
"""
Simple script to help update the TODO.md file with new tasks and progress.
"""

import os
import datetime
from pathlib import Path

def get_current_timestamp():
    """Get current UTC timestamp for TODO updates."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def add_task(priority="MEDIUM", title="", description=""):
    """Add a new task to the TODO file."""
    if not title:
        title = input("Task title: ")
    if not description:
        description = input("Task description: ")
    
    task_entry = f"- [ ] **{title}**\n  - {description}\n"
    
    print(f"\nNew {priority} task:")
    print(task_entry)
    
    # Here you could append to the TODO file
    # For now, just print instructions
    print(f"Add this to the {priority} PRIORITY section in TODO.md")

def mark_completed(task_name=""):
    """Mark a task as completed."""
    if not task_name:
        task_name = input("Task name to mark completed: ")
    
    timestamp = get_current_timestamp()
    print(f"\nMark as completed: {task_name}")
    print(f"Move to ✅ Completed Tasks section with timestamp: ({timestamp.split()[0]})")

def update_timestamp():
    """Update the Last Updated timestamp in TODO.md"""
    timestamp = get_current_timestamp()
    todo_file = Path("TODO.md")
    
    if todo_file.exists():
        content = todo_file.read_text()
        
        # Update the timestamp line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("**Last Updated**:"):
                lines[i] = f"**Last Updated**: {timestamp}"
                break
        
        # Write back
        todo_file.write_text('\n'.join(lines))
        print(f"✅ Updated TODO.md timestamp: {timestamp}")
    else:
        print("❌ TODO.md not found!")

def show_next_tasks():
    """Show the next high priority tasks."""
    print("\n🎯 NEXT HIGH PRIORITY TASKS:")
    print("1. Fix Flux Model Version Issue")
    print("2. Optimize LoRA Integration") 
    print("3. Validate End-to-End Pipeline")
    print("\nSee TODO.md for full details and progress tracking.")

def main():
    """Main menu for TODO management."""
    print("📋 Face The Music - TODO Manager")
    print("=" * 40)
    print("1. Add new task")
    print("2. Mark task completed")
    print("3. Update timestamp")
    print("4. Show next priority tasks")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        priority = input("Priority (HIGH/MEDIUM/LOW): ").upper()
        if priority not in ["HIGH", "MEDIUM", "LOW"]:
            priority = "MEDIUM"
        add_task(priority)
    elif choice == "2":
        mark_completed()
    elif choice == "3":
        update_timestamp()
    elif choice == "4":
        show_next_tasks()
    elif choice == "5":
        print("👋 Happy coding!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
