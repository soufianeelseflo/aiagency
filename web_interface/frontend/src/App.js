// web_interface/frontend/src/App.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

function App() {
    const [agentStatus, setAgentStatus] = useState({});
    const [command, setCommand] = useState('');
    const [selectedAgent, setSelectedAgent] = useState('orchestrator');

    useEffect(() => {
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

    const sendCommand = async () => {
        try {
            const response = await axios.post(`${API_BASE_URL}/send-command`, {
                agent: selectedAgent,
                command: command
            });
            alert(response.data.message);
            setCommand(''); // Clear input
        } catch (error) {
            console.error("Failed to send command:", error);
            alert("Oops! Something went wrong.");
        }
    };

    return (
        <div className="App" style={{ padding: '20px' }}>
            <h1>AI Agency Control Room</h1>
            <div style={{ marginBottom: '20px' }}>
                <h2>Send a Command</h2>
                <select
                    value={selectedAgent}
                    onChange={(e) => setSelectedAgent(e.target.value)}
                    style={{ marginRight: '10px' }}
                >
                    <option value="orchestrator">Orchestrator</option>
                    <option value="voice_agent">Voice Agent</option>
                </select>
                <input
                    type="text"
                    value={command}
                    onChange={(e) => setCommand(e.target.value)}
                    placeholder="e.g., 'call client X' or 'begin'"
                    style={{ width: '300px', marginRight: '10px' }}
                />
                <button onClick={sendCommand}>Send</button>
            </div>
            <div>
                {Object.entries(agentStatus).map(([agent, status]) => (
                    <div key={agent} style={{ border: '1px solid #ccc', padding: '10px', marginBottom: '10px' }}>
                        <h2>{agent}</h2>
                        <pre>{JSON.stringify(status, null, 2)}</pre>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default App;