#!/usr/bin/env python3
"""
Test script to demonstrate the improved UATS tree visualization.
"""

import tempfile
from pathlib import Path
from rely.inference.uats import Branch, _generate_tree_image

def create_sample_branches():
    """Create sample branches for testing tree visualization."""
    branches = []
    
    # Create branches with different step counts and scores
    branches.append(Branch(
        text="Step 1 reasoning...",
        ids=None,  # Not needed for visualization
        step_count=1,
        score=0.85,
        uncertainty=0.3,
        value=0.85,
        total_tokens=150
    ))
    
    branches.append(Branch(
        text="Step 1 alternative reasoning...",
        ids=None,
        step_count=1,
        score=0.72,
        uncertainty=0.6,
        value=0.72,
        total_tokens=120
    ))
    
    branches.append(Branch(
        text="Step 2 continuation...",
        ids=None,
        step_count=2,
        score=0.91,
        uncertainty=0.2,
        value=0.91,
        total_tokens=280
    ))
    
    branches.append(Branch(
        text="Step 2 alternative...",
        ids=None,
        step_count=2,
        score=0.78,
        uncertainty=0.4,
        value=0.78,
        total_tokens=250
    ))
    
    branches.append(Branch(
        text="Step 3 final reasoning...",
        ids=None,
        step_count=3,
        score=0.95,
        uncertainty=0.1,
        value=0.95,
        total_tokens=400
    ))
    
    return branches

def main():
    """Main function to test tree visualization."""
    print("Creating sample UATS branches...")
    branches = create_sample_branches()
    
    print(f"Created {len(branches)} sample branches")
    for i, branch in enumerate(branches):
        print(f"  Branch {i}: Step {branch.step_count}, Score {branch.score:.3f}")
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        print(f"\nGenerating tree visualization in: {output_dir}")
        
        try:
            _generate_tree_image(branches, output_dir, "test_search_tree.png")
            print("✓ Tree visualization generated successfully!")
            print(f"  Output file: {output_dir / 'test_search_tree.png'}")
        except Exception as e:
            print(f"✗ Failed to generate tree visualization: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 