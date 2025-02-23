import React, { useEffect, useState } from 'react';
import axios from 'axios';

// Define the base URL for the backend API
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'https://your-domain.com/api';

function App() {
    const [agentStatus, setAgentStatus] = useState({});

    useEffect(() => {
        // Fetch agent status every 10 seconds
        const interval = setInterval(async () => {
            try {
                const response = await axios.get(`${API_BASE_URL}/agent-status`);
                setAgentStatus(response.data);
            } catch (error) {
                console.error("Failed to fetch agent status:", error);
            }
        }, 10000);

        return () => clearInterval(interval);
    }, []);

    const updateAgent = async (agentName, updates) => {
        try {
            await axios.post(`${API_BASE_URL}/update-agent`, { agent: agentName, updates });
            alert(`${agentName} updated successfully.`);
        } catch (error) {
            console.error("Failed to update agent:", error);
        }
    };

    return (
        <div className="App">
            <h1>AI Agency Dashboard</h1>
            <div>
                {Object.entries(agentStatus).map(([agent, status]) => (
                    <div key={agent}>
                        <h2>{agent}</h2>
                        <pre>{JSON.stringify(status, null, 2)}</pre>
                        <button onClick={() => updateAgent(agent, { priority: "high" })}>
                            Set High Priority
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default App;