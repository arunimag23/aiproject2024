import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const result = await axios.get('http://127.0.0.1:5000/api/data');
      setData(result.data);
    };

    fetchData();
  }, []);

  return (
    <div className="App">
      <h1>React and Flask Integration</h1>
      {data && <p>{data.message}</p>}
    </div>
  );
}

export default App;
