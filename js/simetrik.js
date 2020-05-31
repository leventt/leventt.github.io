var fftSteps = 128
var fft = new Tone.FFT(fftSteps).toMaster()

var player = new Tone.Player('res/simetrik/triplet.m4a').connect(fft)
player.autostart = true
player.loop = true

function setup() {
    var canvas = createCanvas(window.innerWidth, window.innerHeight)
    canvas.parent("simetrikContainer")

    font = loadFont('res/cubic/cubic.ttf')
    textFont(font, 16.)
}

function simetri(angle, x, y)
{
    var s = sin(angle)
    var c = cos(angle)

    x -= width / 2.
    y -= height / 2.

    var sX = x * c - y * s
    var sY = x * s + y * c

    return [sX + width / 2., sY + height / 2.]
}

function draw() {
    background(39, 39, 39, 12)

    var R = 65
    var amplitude = 2.

    stroke(127, 255, 212)
    strokeWeight(8.)

    noFill()
    beginShape()
    var spectrum = fft.getValue();
    var angle = 0.
    var spectrumVal = 0.
    var x, y;
    for (var i = 0; i < fftSteps * 8; i++) {
        angle = (2 * PI * i) / fftSteps
        spectrumVal = spectrum[i % 8]
        if (isNaN(spectrumVal) || spectrumVal == Infinity) {
            spectrumVal = 0.
        }
        x = (R + amplitude * sin(spectrumVal / 8.)) * cos(angle) + width / 2.
        y = (R + amplitude * sin(spectrumVal / 8.)) * sin(angle) + height / 2.
        vertex(x, y)
    }
    endShape(CLOSE)

    textAlign(CENTER)
    textSize(16.)
    stroke(0)
    fill(255)
    strokeWeight(1.)

    textSize(32.)
    text('draw.', width / 2., 75.)

    // DRAW
    if (mouseIsPressed) {
        stroke(127, 255, 212)
        strokeWeight(4.)
        noFill()
        beginShape()
        vertex(pmouseX, pmouseY)
        vertex(mouseX, mouseY)
        endShape(CLOSE)
        for (var i = 1; i < 8; i++) {
            p = simetri((2 * PI / 8.) * i, pmouseX, pmouseY)
            c = simetri((2 * PI / 8.) * i, mouseX, mouseY)
            beginShape()
            vertex(p[0], p[1])
            vertex(c[0], c[1])
            endShape(CLOSE)
        }
    }
}

function mousePressed() {

}

function mouseReleased() {
    
}