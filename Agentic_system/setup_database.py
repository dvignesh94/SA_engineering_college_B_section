import asyncio
import asyncpg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_database():
    """Setup database tables for the AI system"""
    
    # Your Neon PostgreSQL connection
    database_url = "postgresql://neondb_owner:npg_wB7SZVOPy0cs@ep-late-mountain-adnikfry-pooler.c-2.us-east-1.aws.neon.tech/neondb"
    
    try:
        # Connect to database
        conn = await asyncpg.connect(database_url)
        logger.info("Connected to Neon PostgreSQL database")
        
        # Check if sensor_data table exists and add columns if needed
        result = await conn.fetch("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'sensor_data'
        """)
        
        existing_columns = [row['column_name'] for row in result]
        
        if 'sensor_data' not in [row['table_name'] for row in await conn.fetch("SELECT table_name FROM information_schema.tables WHERE table_name = 'sensor_data'")]:
            # Create new sensor_data table
            await conn.execute("""
                CREATE TABLE sensor_data (
                    id SERIAL PRIMARY KEY,
                    reading_time TIMESTAMP DEFAULT NOW(),
                    sensor_value FLOAT,
                    agent_id VARCHAR(50),
                    ai_decision TEXT,
                    led_status BOOLEAN DEFAULT FALSE
                )
            """)
            logger.info("Created new sensor_data table")
        else:
            # Add missing columns to existing table
            if 'agent_id' not in existing_columns:
                await conn.execute("ALTER TABLE sensor_data ADD COLUMN agent_id VARCHAR(50)")
                logger.info("Added agent_id column")
            
            if 'ai_decision' not in existing_columns:
                await conn.execute("ALTER TABLE sensor_data ADD COLUMN ai_decision TEXT")
                logger.info("Added ai_decision column")
            else:
                # Update existing column to TEXT if it's VARCHAR(100)
                await conn.execute("ALTER TABLE sensor_data ALTER COLUMN ai_decision TYPE TEXT")
                logger.info("Updated ai_decision column to TEXT")
            
            if 'led_status' not in existing_columns:
                await conn.execute("ALTER TABLE sensor_data ADD COLUMN led_status BOOLEAN DEFAULT FALSE")
                logger.info("Added led_status column")
            
            logger.info("Updated existing sensor_data table")
        
        # Drop and recreate agent_commands table to ensure correct structure
        await conn.execute("DROP TABLE IF EXISTS agent_commands")
        await conn.execute("""
            CREATE TABLE agent_commands (
                id SERIAL PRIMARY KEY,
                agent_id VARCHAR(50),
                command VARCHAR(100),
                timestamp TIMESTAMP DEFAULT NOW(),
                status VARCHAR(20) DEFAULT 'pending',
                ai_reasoning TEXT
            )
        """)
        logger.info("Created agent_commands table")
        
        # Create indexes for better performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensor_data_agent_id 
            ON sensor_data(agent_id)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensor_data_reading_time 
            ON sensor_data(reading_time)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_commands_agent_id 
            ON agent_commands(agent_id)
        """)
        
        logger.info("Created database indexes")
        
        # Insert sample data
        await conn.execute("""
            INSERT INTO sensor_data (sensor_value, agent_id, ai_decision, led_status)
            VALUES (15.5, 'ESP32_001', 'Safe distance detected', false)
        """)
        
        await conn.execute("""
            INSERT INTO agent_commands (agent_id, command, ai_reasoning, status)
            VALUES ('ESP32_001', 'LED_OFF', 'Object at safe distance', 'completed')
        """)
        
        logger.info("Inserted sample data")
        
        await conn.close()
        logger.info("Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(setup_database())
