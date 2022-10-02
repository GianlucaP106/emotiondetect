import AudioRecorder from "./components/AudioRecorder.js";
import "./styles/App.css";
import React, { useState } from "react";
import "./styles/AudioStyles.css";
const axios = require("axios");

function App() {
  const [blobSrc, setBlobsrc] = useState(null);

  const setRecorderBlobsrc = (wtvSrc) => {
    setBlobsrc(wtvSrc);
  };

  async function sendFile() {
    // const file = "/hello.wav"
    // const fd = new FormData();
    // fd.set('file', file)

    // try {
    //   const response = await fetch("http://127.0.0.1:5000/audio", {
    //     mode: 'no-cors',
    //     method: "POST",
    //     body: fd
    //   });
    // } catch (err) {
    //   console.log(err);
    // }

    // var audioFile = fs.createReadStream("/Users/parsalangari/Desktop/SIDE_LEARNING/hack-repo/emotiondetecttemp/front/public/hello.wav")
    var form = new FormData();
    const currentSrc = blobSrc;
    console.log(currentSrc);
    console.log(typeof(currentSrc))
    form.append('file', currentSrc , 'file')

    await axios.post("http://127.0.0.1:5000/audio",
        form,
        {
            headers: {
                "Content-Type": "multipart/form-data"
            }
        }
    )
    .then(result => {
        console.log(result.data)
    })
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
