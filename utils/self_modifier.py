import os
import psycopg2
import ast
import subprocess
from datetime import datetime

class CodeRefactorer:
    def __init__(self):
        self.db_conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host='postgresql'
        )
        self.db_conn.autocommit = True

    def safe_apply_update(self, update_plan: dict) -> None:
        """Deploy code changes safely with PostgreSQL logging"""
        try:
            # Validate syntax
            ast.parse(update_plan['new_content'])
            
            # Log update plan to PostgreSQL
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO code_updates (file_path, new_content, timestamp)
                    VALUES (%s, %s, %s)
                """, (
                    update_plan['file_path'],
                    update_plan['new_content'],
                    datetime.utcnow()
                ))
            
            # Apply changes and run tests
            with open(update_plan['file_path'], 'w') as f:
                f.write(update_plan['new_content'])
            subprocess.run(["pytest"], check=True)
            
            # Commit changes to Git
            subprocess.run(["git", "add", update_plan['file_path']], check=True)
            subprocess.run(["git", "commit", "-m", update_plan['commit_message']], check=True)
        except Exception as e:
            # Rollback changes
            subprocess.run(["git", "reset", "--hard", "HEAD"], check=True)
            raise RuntimeError(f"Update failed: {str(e)}")