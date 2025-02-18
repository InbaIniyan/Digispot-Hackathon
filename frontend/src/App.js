import React, { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, Title, Tooltip, Legend, CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement } from 'chart.js';
import './styles.css';

// Register chart components
ChartJS.register(Title, Tooltip, Legend, CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement);

function App() {
  const [websites, setWebsites] = useState('');
  const [modelName, setModelName] = useState('MiniLM');
  const [summary, setSummary] = useState([]);
  const [detailed, setDetailed] = useState([]);
  const [chartData, setChartData] = useState(null);

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      x: {
        grid: {
          color: '#e0e0e0', // Light grey for the grid
        },
      },
      y: {
        grid: {
          color: '#e0e0e0', // Light grey for the grid
        },
      },
    },
    layout: {
      padding: 20,
    },
    elements: {
      line: {
        tension: 0.4,
      },
    },
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const websiteArray = websites.split('\n').map((url) => url.trim());

    try {
      const response = await axios.post('http://localhost:5000/generate_link_suggestions', {
        websites: websiteArray,
        model_name: modelName,
      });

      setSummary(response.data.summary);
      setDetailed(response.data.detailed);

      // Prepare data for chart
      const websitesNames = response.data.summary.map(row => row.Website);
      const clustersData = response.data.summary.map(row => row.Clusters);
      const suggestionsData = response.data.summary.map(row => row['Total Suggestions']);

      setChartData({
        labels: websitesNames,
        datasets: [
          {
            label: 'Number of Clusters',
            data: clustersData,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: true,
          },
          {
            label: 'Total Suggestions',
            data: suggestionsData,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            fill: true,
          },
        ],
      });
    } catch (error) {
      console.error('Error submitting the URLs:', error);
    }
  };

  return (
    <div className="App">
      <h1>Internal Linking Tool</h1>

      <form onSubmit={handleSubmit} className="form-container">
        <div className="form-group">
          <label htmlFor="websites">Enter URLs (one per line):</label>
          <textarea
            id="websites"
            value={websites}
            onChange={(e) => setWebsites(e.target.value)}
            rows="5"
            className="input-field"
          />
        </div>

        <div className="form-group">
          <label htmlFor="modelName">Select Model:</label>
          <select
            id="modelName"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            className="input-field"
          >
            <option value="MiniLM">MiniLM</option>
            <option value="BERT">BERT</option>
            <option value="T5">T5</option>
          </select>
        </div>

        <button type="submit" className="submit-btn">Generate Suggestions</button>
      </form>

      <h2>Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Website</th>
            <th>Model</th>
            <th>Pages Processed</th>
            <th>Clusters</th>
            <th>Total Suggestions</th>
            <th>Avg Cluster Size</th>
            <th>Silhouette Score</th>
          </tr>
        </thead>
        <tbody>
          {summary.map((row, index) => (
            <tr key={index}>
              <td>{row.Website}</td>
              <td>{row.Model}</td>
              <td>{row['Pages Processed']}</td>
              <td>{row.Clusters}</td>
              <td>{row['Total Suggestions']}</td>
              <td>{row['Avg Cluster Size']}</td>
              <td>{row['Silhouette Score']}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {chartData && (
        <div>
          <h2>Visualization of Clusters & Suggestions</h2>
          <Line data={chartData} options={chartOptions} />
        </div>
      )}

      <h2>Detailed Suggestions</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Source Page</th>
            <th>Target Page</th>
            <th>Suggested Anchor Text</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {detailed.map((row, index) => (
            <tr key={index}>
              <td>{row.Model}</td>
              <td>{row['Source Page']}</td>
              <td>{row['Target Page']}</td>
              <td>{row['Suggested Anchor Text']}</td>
              <td>{row.Reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
