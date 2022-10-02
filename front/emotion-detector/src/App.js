
import './styles/App.css';
import './styles/AudioStyles.css';

import AudioRecorder from "./components/AudioRecorder.js";

function App() {

  async function sendFile() {
    const file = "/hello.wav"
    const fd = new FormData();
    fd.set('file', file)

    try {
      const response = await fetch("http://localhost:3001", {
        method: "POST",
        body: fd
      });
    } catch (err) {
      console.log(err);
    }

  }

  return (
    <div className="App">
      <button onClick={sendFile}>send</button>
      <h1>hello</h1>
      <AudioRecorder />
    </div>
  );
}

export default App;
