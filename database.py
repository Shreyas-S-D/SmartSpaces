import sqlite3
import json
import os
from datetime import datetime
import numpy as np

def init_db(db_path='smartspaces.db'):
    """Initialize SQLite database"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create plans table
    c.execute('''
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            requirements TEXT NOT NULL,
            plans TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create user feedback table
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            plan_id INTEGER,
            rating INTEGER,
            comments TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (plan_id) REFERENCES plans (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

def save_user_plan(user_id, plans, requirements, db_path='smartspaces.db'):
    """Save generated plans to database"""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Convert plans to JSON-serializable format with proper type conversion
        plans_serializable = []
        for plan in plans:
            # Convert numpy array to list and ensure all values are Python native types
            if hasattr(plan, 'tolist'):
                plan_list = plan.tolist()
            else:
                plan_list = plan
            
            # Ensure all numbers in the list are Python native types
            if isinstance(plan_list, list):
                plan_list = convert_to_native_types(plan_list)
            
            plans_serializable.append(plan_list)
        
        # Convert requirements to ensure native types
        requirements_fixed = convert_to_native_types(requirements)
        
        c.execute('''
            INSERT INTO plans (user_id, requirements, plans)
            VALUES (?, ?, ?)
        ''', (user_id, json.dumps(requirements_fixed), json.dumps(plans_serializable)))
        
        plan_id = c.lastrowid
        conn.commit()
        conn.close()
        
        print(f"Saved plan {plan_id} for user {user_id}")
        return plan_id
        
    except Exception as e:
        print(f"Error saving plan: {e}")
        return None

def convert_to_native_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    else:
        return obj

def get_user_plans(user_id, db_path='smartspaces.db'):
    """Retrieve user's previous plans"""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT id, requirements, plans, created_at 
            FROM plans 
            WHERE user_id = ? 
            ORDER BY created_at DESC
            LIMIT 10
        ''', (user_id,))
        
        results = c.fetchall()
        conn.close()
        
        plans = []
        for row in results:
            plan_id, requirements_json, plans_json, created_at = row
            plans.append({
                'id': plan_id,
                'requirements': json.loads(requirements_json),
                'plans': json.loads(plans_json),
                'created_at': created_at
            })
        
        return plans
        
    except Exception as e:
        print(f"Error retrieving plans: {e}")
        return []

def save_feedback(user_id, plan_id, rating, comments=None, db_path='smartspaces.db'):
    """Save user feedback for plans"""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO feedback (user_id, plan_id, rating, comments)
            VALUES (?, ?, ?, ?)
        ''', (user_id, plan_id, rating, comments))
        
        conn.commit()
        conn.close()
        
        print(f"Saved feedback for plan {plan_id}")
        return True
        
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False