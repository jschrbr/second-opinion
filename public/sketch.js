let video; // Create a KNN classifier
const knnClassifier = ml5.KNNClassifier();
let featureExtractor;

async function preload() {
  featureExtractor = ml5.featureExtractor("MobileNet", modelReady);
  const KNNclass = window.localStorage.getItem("KNNclass");
  const KNNobj = JSON.parse(KNNclass);
  // console.log(KNNobj);
  await knnClassifier.load(KNNobj);
}

function setup() {
  video = createCapture(VIDEO);
  video.hide();
  createButtons();
}

function draw() {
  drawingContext.shadowOffsetX = 5;
  drawingContext.shadowOffsetY = -5;
  video.parent("videoContainer");
  wid = windowWidth * 0.8;
  video.size(wid, wid);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

function modelReady() {
  select(".loader").hide();
  video.show();
  classify();
}

// Add the current frame from the video to the classifier
function addExample(label) {
  console.log(`Added ${label}`);
  const features = featureExtractor.infer(video);
  knnClassifier.addExample(features, label);
  classify();
}
// Predict the current frame.
async function classify() {
  const numLabels = knnClassifier.getNumLabels();
  if (numLabels <= 0) {
    console.error("There is no examples in any label");
    return;
  }
  const dataset = await knnClassifier.getClassifierDataset();

  Object.keys(dataset).forEach((key) => {
    dataset[key].label = knnClassifier.mapStringToIndex[key];
    dataset[key].kept = false;
  });
  const tensors = Object.keys(dataset).map((key) => {
    const t = dataset[key];
    if (t) {
      return t.dataSync();
    }
    return null;
  });
  window.localStorage.setItem("KNNclass", JSON.stringify({ dataset, tensors }));
  const features = featureExtractor.infer(video);
  const res = await knnClassifier.classify(features);
  gotResults(null, res);
}
// A util function to create UI buttons
function createButtons() {
  // When the A button is pressed, add the current frame
  // from the video with a label of "rock" to the classifier
  buttonLike = select("#addClassLike");
  buttonLike.mousePressed(function () {
    let like = setInterval(() => {
      addExample("Like");
    }, 300);
    buttonLike.mouseReleased(() => clearInterval(like));
  });
  buttonLike.touchStarted(function () {
    let like = setInterval(() => {
      addExample("Like");
    }, 300);
    buttonLike.touchEnded(() => clearInterval(like));
  });

  buttonNoLike = select("#addClassNoLike");
  buttonNoLike.mousePressed(function () {
    let like = setInterval(() => {
      addExample("NoLike");
    }, 300);
    buttonLike.mouseReleased(() => clearInterval(like));
  });

  buttonNoLike.touchStarted(function () {
    let like = setInterval(() => {
      addExample("NoLike");
    }, 300);
    buttonLike.touchEnded(() => clearInterval(like));
  });

  buttonClearAll = select("#clearAll");
  buttonClearAll.mousePressed(clearAllLabels);
}

function gotResults(err, result) {
  // Display any error
  if (err) {
    console.error(err);
  }

  if (result.confidencesByLabel) {
    const confidences = result.confidencesByLabel;
    const labels = Object.keys(confidences);
    // result.label is the label that has the highest confidence
    id = result.label;
    if (result.label) {
      select("#result").html(labels[id]);
      select("#confidence").html(
        `${(result.confidences[result.label] * 100).toFixed(2)} %`
      );
    }
  }
  classify();
}

// Clear all the examples in all labels
function clearAllLabels() {
  knnClassifier.clearAllLabels();
  window.localStorage.clear();
  select("#result").html("...");
  select("#confidence").html("...");
}