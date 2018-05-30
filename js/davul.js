var play = false
var movement = 0.
var fftSteps = 128
var bpm = 96

var fft = new Tone.FFT(fftSteps).toMaster()
var crusher = new Tone.BitCrusher(6).connect(fft)

var closedHiHat = new Tone.Player('res/davul/hat.wav').connect(crusher)

var wasKicked = false

function triggerHat(time) {
    closedHiHat.start(time)
}

var kick = new Tone.Player('res/davul/kick.wav').connect(crusher)

function triggerKick(time) {
    kick.start(time)
    wasKicked = true
}

var snare = new Tone.Player('res/davul/snare.wav').connect(crusher)

function triggerSnare(time) {
    snare.start(time)
    wasKicked = false
}

// [[hat], [snare], [kick]]
var times = [['0:0:0', '0:2:0', '1:0:0', '1:2:0', '2:0:0', '2:2:0', '3:0:0', '3:2:0'], ['1:0:0', '3:0:0'], ['0:0:0', '2:2:0']]
var eventIds = [[], [], []]
function schedule() {
    times[0].forEach(t => {
        id = Tone.Transport.schedule(triggerHat, t)
        eventIds[0].push(id)
    })

    times[1].forEach(t => {
        id = Tone.Transport.schedule(triggerSnare, t)
        eventIds[1].push(id)
    })

    times[2].forEach(t => {
        id = Tone.Transport.schedule(triggerKick, t)
        eventIds[2].push(id)
    })
}

function setup() {
    var canvas = createCanvas(window.innerWidth, window.innerHeight)
    canvas.parent("davulContainer")

    schedule()

    Tone.Transport.loopEnd = '4m'
    Tone.Transport.loop = true
    Tone.Transport.bpm.value = bpm *4.

    font = loadFont('res/cubic/cubic.ttf')
    textFont(font, 16.)
}

function drawCommon(size, increment, strokeR, strokeG, strokeB, sWeight) {
    stroke(strokeR, strokeG, strokeB)
    strokeWeight(sWeight)
    fill(49)
    ellipse(width / 2, height / 2, size, size)
}

function drawHats(size, increment) {
    drawCommon(size, increment, 255, 255, 0, 2.)
    strokeWeight(12.)
    var half = PI / 32.
    times[0].forEach(t => {
        var offset = (parseInt(t[0]) + parseInt(t[2]) * .25 + parseInt(t[4]) * .125) * PI / 2.
        arc(width / 2, height / 2, size, size, increment + offset - half, increment + offset + half) 
    })
}

function drawSnares(size, increment) {
    drawCommon(size, increment, 127, 255, 212, 2.)
    strokeWeight(12.)
    var half = PI / 32.
    times[1].forEach(t => {
        var offset = (parseInt(t[0]) + parseInt(t[2]) * .25 + parseInt(t[4]) * .125) * PI / 2.
        arc(width / 2, height / 2, size, size, increment + offset - half, increment + offset + half)
    })
}


function drawKicks(size, increment) {
    drawCommon(size, increment, 255, 0, 255, 2.)
    strokeWeight(12.)
    var half = PI / 32.
    times[2].forEach(t => {
        var offset = (parseInt(t[0]) + parseInt(t[2]) * .25 + parseInt(t[4]) * .125) * PI / 2.
        arc(width / 2, height / 2, size, size, increment + offset - half, increment + offset + half)
    })
}

function drawCursor(size, increment) {
    drawCommon(size, increment, 255, 255, 255, .5)
    strokeWeight(5.)
    var half = PI / 64.
    arc(width / 2, height / 2, size, size, increment - half, increment + half)
}

grid16Times = [0., 0.25, .5, .75,  1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75]

function drawGrid(size, increment) {
    drawCommon(size, increment, 150, 150, 150, .1)
    strokeWeight(1.)
    var half = PI / 32.
    grid16Times.forEach(t => {
        var offset = t * PI / 2.
        arc(width / 2, height / 2, size, size, increment + offset - half, increment + offset + half)
    })
}

var meter = ['2', 'e', 'n', 'a', '3', 'e', 'n', 'a', '4', 'e', 'n', 'a', '1', 'e', 'n', 'a']
function draw() {
    background(39)

    movement = Tone.Transport.progress * PI * 2 - (PI / 2.)
    drawGrid(325, -1 * PI / 2.)
    drawCursor(320, movement)
    drawHats(300, -1 * PI / 2.)
    drawCursor(280, movement)
    drawGrid(275, -1 * PI / 2.)
    drawGrid(225, -1 * PI / 2.)
    drawCursor(220, movement)
    drawSnares(200, -1 * PI / 2.)
    drawCursor(180, movement)
    drawGrid(175, -1 * PI / 2.)
    drawGrid(125, -1 * PI / 2.)
    drawCursor(120, movement)
    drawKicks(100, -1 * PI / 2.)
    drawCursor(80, movement)
    drawGrid(75, -1 * PI / 2.)


    var R = 175
    var theta = 0
    var n = 3.
    var amplitude = 2.

    if (wasKicked) {
        stroke(255, 0, 255)
    }
    else {
        stroke(127, 255, 212)
    }
    if (!play) {
        stroke(127, 127, 127)
    }
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
        if (isNaN(spectrumVal) || spectrumVal == Infinity || !play) {
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
    text(bpm.toString(), width / 2., height / 2. + 8)

    textSize(32.)
    text('davul.', width / 2., 75.)

    for (var i = 0; i < 16; i++) {
        angle = (2 * PI * i) / 16
        spectrumVal = spectrum[i % 8]
        if (isNaN(spectrumVal) || spectrumVal == Infinity || !play) {
            spectrumVal = 0.
        }
        x = (R + 10) * cos(angle) + width / 2.
        y = (R + 10) * sin(angle) + height / 2.
        textSize(8.)
        if (i % 4 != 0) {
            textSize(5.)
        }
        stroke(0)
        fill(255)
        strokeWeight(1.)
        text(meter[i], x, y + 3)
    }
}

function mouseClicked() {
    var dx = (width / 2) - mouseX
    var dy = (height / 2) - mouseY
    var clickR = dist(width / 2, height / 2, mouseX, mouseY)
    var clickT = -1 * atan2(dx, dy)

    var t = measure(clickT)
    if (clickR <= 25) {
        play = !play
        if (play) {
            Tone.Transport.start()
        } else {
            Tone.Transport.pause()
        }
    } else if (clickR <= 75) {
        index = times[2].indexOf(t)
        if (index == -1) {
            id = Tone.Transport.schedule(triggerKick, t)
            times[2].push(t)
            eventIds[2].push(id)
        }
        else {
            Tone.Transport.clear(eventIds[2][index])
            times[2].splice(index, 1)
            eventIds[2].splice(index, 1)
        }
    } else if (clickR <= 125) {
        index = times[1].indexOf(t)
        if (index == -1) {
            id = Tone.Transport.schedule(triggerSnare, t)
            times[1].push(t)
            eventIds[1].push(id)
        }
        else {
            Tone.Transport.clear(eventIds[1][index])
            times[1].splice(index, 1)
            eventIds[1].splice(index, 1)
        }
    } else if (clickR <= 175) {
        index = times[0].indexOf(t)
        if (index == -1) {
            id = Tone.Transport.schedule(triggerHat, t)
            times[0].push(t)
            eventIds[0].push(id)
        }
        else {
            Tone.Transport.clear(eventIds[0][index])
            times[0].splice(index, 1)
            eventIds[0].splice(index, 1)
        }
    } else if (clickR > 200) {
        if (mouseX > (width / 2.)) {
            bpm += 1
        }
        else {
            bpm -= 1
        }
        Tone.Transport.bpm.value = bpm * 4.
    }
}

function doubleClicked() {
    var dx = (width / 2) - mouseX
    var dy = (height / 2) - mouseY
    var clickR = dist(width / 2, height / 2, mouseX, mouseY)
    var clickT = -1 * atan2(dx, dy)

    if (clickR <= 25) {
        Tone.Transport.stop()
        play = false
        bpm = 96
        Tone.Transport.bpm.value = bpm * 4.
    }
}

var leftMap = ['0:0:0', '3:3:0', '3:2:0', '3:1:0', '3:0:0', '2:3:0', '2:2:0', '2:1:0', '2:0:0']
var rightMap = ['0:0:0', '0:1:0', '0:2:0', '0:3:0', '1:0:0', '1:1:0', '1:2:0', '1:3:0', '2:0:0']
function measure(clickT) {
    var i
    if (clickT < 0) {
        i = Math.round(lerp(0, 8, abs(clickT / PI)))
        i = leftMap[i]
    } else {
        i = Math.round(lerp(0, 8, clickT / PI))
        i = rightMap[i]
    }
    return i
}