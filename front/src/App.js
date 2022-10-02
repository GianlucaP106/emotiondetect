import AudioRecorder from "./components/AudioRecorder.js";
import "./styles/App.css";

import React, { useState } from "react";
import { saveAs } from "file-saver";
import "./styles/AudioStyles.css";
const os = require("os");
// const axios = require("axios");

function App() {
  const [blobSrc, setBlobsrc] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);

  const setRecorderBlobsrc = (wtvSrc) => {
    setBlobsrc(wtvSrc);
  };

  async function sendFile() {
    const currentSrc = blobSrc;
    let filename = "audioFileMLMAIS-" + String(currentIndex);
    setCurrentIndex(currentIndex + 1);
    saveAs(currentSrc, filename, "audioFileMLMAIS-" + String(currentIndex));
    filename = "audioFileMLMAIS-" + String(currentIndex);
    const dirname = "/Users/my-mac/Downloads";
    const filepath = os.path.join(dirname, filename);

    await fetch("http://127.0.0.1:5000/audio ", {
      method: "POST",
      header: {
        "Content-Type": "application/json",
      },
      body: filepath,
      cache: "default",
    }).then((result) => {
      console.log(`Emotion of your text: ${result.json()}`);
    });
  }

  return (
    <div className="App">
      <button onClick={sendFile}>send</button>
      <h1>hello</h1>
      <AudioRecorder setAppSrc={setRecorderBlobsrc} />
    </div>
  );
}

export default App;
