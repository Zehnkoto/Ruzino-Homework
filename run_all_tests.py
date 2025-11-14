#!/usr/bin/env python3
"""
Test Runner Script for Ruzino Project

This script:
1. Recursively finds all 'tests/' folders under './source/'
2. Runs pytest on any 'test_*.py' files found
3. Runs corresponding '*_test.exe' files from './Binaries/Debug/' for any '.cpp' files
4. Logs all results to './Binaries/Debug/test_log_<gittag>_<timestamp>.log'
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path
from typing import List, Tuple


def get_git_tag() -> str:
    """Get current git tag or commit hash."""
    try:
        # Try to get the current tag
        result = subprocess.run(
            ['git', 'describe', '--tags', '--exact-match'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        
        # If no exact tag, get short commit hash
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"Warning: Could not get git tag: {e}")
    
    return "unknown"


def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def find_test_directories(source_dir: Path) -> List[Path]:
    """Find all 'tests/' directories under source_dir recursively."""
    test_dirs = []
    for root, dirs, _ in os.walk(source_dir):
        if 'tests' in dirs:
            test_dirs.append(Path(root) / 'tests')
    return test_dirs


def find_python_test_files(test_dir: Path) -> List[Path]:
    """Find all test_*.py files in the given directory."""
    return list(test_dir.glob('test_*.py'))


def find_cpp_test_files(test_dir: Path) -> List[Path]:
    """Find all *.cpp files in the given directory."""
    return list(test_dir.glob('*.cpp'))


def cpp_to_exe_name(cpp_file: Path) -> str:
    """Convert cpp filename to expected exe name.
    
    For example:
    - some_file.cpp -> some_file_test.exe
    - renderer.cpp -> renderer_test.exe
    """
    base_name = cpp_file.stem
    if base_name.endswith('_test'):
        return f"{base_name}.exe"
    else:
        return f"{base_name}_test.exe"


def run_pytest(test_dir: Path, log_file) -> Tuple[int, int]:
    """Run pytest in the given directory.
    
    Returns:
        Tuple of (passed, failed) test counts
    """
    python_tests = find_python_test_files(test_dir)
    
    if not python_tests:
        return 0, 0
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Running pytest in: {test_dir}\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()
    
    passed = 0
    failed = 0
    
    for test_file in python_tests:
        log_file.write(f"\n--- Running: {test_file.name} ---\n")
        log_file.flush()
        
        try:
            result = subprocess.run(
                ['pytest', str(test_file), '-v', '--tb=short'],
                capture_output=True,
                text=True,
                cwd=test_dir,
                timeout=300  # 5 minute timeout per test file
            )
            
            log_file.write(result.stdout)
            log_file.write(result.stderr)
            log_file.flush()
            
            if result.returncode == 0:
                log_file.write(f"✓ PASSED: {test_file.name}\n")
                passed += 1
            else:
                log_file.write(f"✗ FAILED: {test_file.name} (exit code: {result.returncode})\n")
                failed += 1
                
        except subprocess.TimeoutExpired:
            log_file.write(f"✗ TIMEOUT: {test_file.name}\n")
            failed += 1
        except Exception as e:
            log_file.write(f"✗ ERROR: {test_file.name} - {str(e)}\n")
            failed += 1
    
    return passed, failed


def run_cpp_tests(test_dir: Path, binaries_dir: Path, log_file) -> Tuple[int, int]:
    """Run C++ test executables corresponding to cpp files in test_dir.
    
    Returns:
        Tuple of (passed, failed) test counts
    """
    cpp_files = find_cpp_test_files(test_dir)
    
    if not cpp_files:
        return 0, 0
    
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"Running C++ tests from: {test_dir}\n")
    log_file.write(f"{'='*80}\n")
    log_file.flush()
    
    passed = 0
    failed = 0
    
    for cpp_file in cpp_files:
        exe_name = cpp_to_exe_name(cpp_file)
        exe_path = binaries_dir / exe_name
        
        log_file.write(f"\n--- Running: {exe_name} (from {cpp_file.name}) ---\n")
        log_file.flush()
        
        if not exe_path.exists():
            log_file.write(f"⚠ SKIPPED: {exe_name} not found in {binaries_dir}\n")
            continue
        
        try:
            result = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                text=True,
                cwd=binaries_dir,
                timeout=300  # 5 minute timeout per test
            )
            
            log_file.write(result.stdout)
            log_file.write(result.stderr)
            log_file.flush()
            
            if result.returncode == 0:
                log_file.write(f"✓ PASSED: {exe_name}\n")
                passed += 1
            else:
                log_file.write(f"✗ FAILED: {exe_name} (exit code: {result.returncode})\n")
                failed += 1
                
        except subprocess.TimeoutExpired:
            log_file.write(f"✗ TIMEOUT: {exe_name}\n")
            failed += 1
        except Exception as e:
            log_file.write(f"✗ ERROR: {exe_name} - {str(e)}\n")
            failed += 1
    
    return passed, failed


def main():
    """Main test runner."""
    # Setup paths
    script_dir = Path(__file__).parent
    source_dir = script_dir / 'source'
    binaries_dir = script_dir / 'Binaries' / 'Debug'
    
    # Create log file
    git_tag = get_git_tag()
    timestamp = get_timestamp()
    log_filename = f"test_log_{git_tag}_{timestamp}.log"
    log_path = binaries_dir / log_filename
    
    # Ensure binaries directory exists
    binaries_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting test run...")
    print(f"Git tag: {git_tag}")
    print(f"Timestamp: {timestamp}")
    print(f"Log file: {log_path}")
    print(f"Searching for tests in: {source_dir}")
    print("-" * 80)
    
    # Find all test directories
    test_dirs = find_test_directories(source_dir)
    print(f"Found {len(test_dirs)} test directories")
    
    # Run all tests
    total_pytest_passed = 0
    total_pytest_failed = 0
    total_cpp_passed = 0
    total_cpp_failed = 0
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Test Run Report\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Git Tag: {git_tag}\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Source Directory: {source_dir}\n")
        log_file.write(f"Binaries Directory: {binaries_dir}\n")
        log_file.write(f"{'='*80}\n\n")
        log_file.flush()
        
        for test_dir in test_dirs:
            print(f"\nProcessing: {test_dir.relative_to(script_dir)}")
            
            # Run Python tests
            pytest_passed, pytest_failed = run_pytest(test_dir, log_file)
            total_pytest_passed += pytest_passed
            total_pytest_failed += pytest_failed
            
            if pytest_passed + pytest_failed > 0:
                print(f"  Python tests: {pytest_passed} passed, {pytest_failed} failed")
            
            # Run C++ tests
            cpp_passed, cpp_failed = run_cpp_tests(test_dir, binaries_dir, log_file)
            total_cpp_passed += cpp_passed
            total_cpp_failed += cpp_failed
            
            if cpp_passed + cpp_failed > 0:
                print(f"  C++ tests: {cpp_passed} passed, {cpp_failed} failed")
        
        # Write summary
        log_file.write(f"\n\n{'='*80}\n")
        log_file.write(f"TEST SUMMARY\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Python Tests:\n")
        log_file.write(f"  Passed: {total_pytest_passed}\n")
        log_file.write(f"  Failed: {total_pytest_failed}\n")
        log_file.write(f"  Total:  {total_pytest_passed + total_pytest_failed}\n")
        log_file.write(f"\n")
        log_file.write(f"C++ Tests:\n")
        log_file.write(f"  Passed: {total_cpp_passed}\n")
        log_file.write(f"  Failed: {total_cpp_failed}\n")
        log_file.write(f"  Total:  {total_cpp_passed + total_cpp_failed}\n")
        log_file.write(f"\n")
        log_file.write(f"Overall:\n")
        log_file.write(f"  Passed: {total_pytest_passed + total_cpp_passed}\n")
        log_file.write(f"  Failed: {total_pytest_failed + total_cpp_failed}\n")
        log_file.write(f"  Total:  {total_pytest_passed + total_pytest_failed + total_cpp_passed + total_cpp_failed}\n")
        log_file.write(f"{'='*80}\n")
    
    # Print summary to console
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Python Tests: {total_pytest_passed} passed, {total_pytest_failed} failed")
    print(f"C++ Tests:    {total_cpp_passed} passed, {total_cpp_failed} failed")
    print(f"Overall:      {total_pytest_passed + total_cpp_passed} passed, "
          f"{total_pytest_failed + total_cpp_failed} failed")
    print("="*80)
    print(f"\nDetailed log saved to: {log_path}")
    
    # Exit with error code if any tests failed
    if total_pytest_failed + total_cpp_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
