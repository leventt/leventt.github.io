(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
window.SG = require("ml-savitzky-golay-generalized");

},{"ml-savitzky-golay-generalized":2}],2:[function(require,module,exports){
'use strict';

/**
 * Apply Savitzky Golay algorithm
 * @param {array} [ys] Array of y values
 * @param {array|number} [xs] Array of X or deltaX
 * @param {object} [options={}]
 * @param {number} [options.windowSize=9]
 * @param {number} [options.derivative=0]
 * @param {number} [options.polynomial=3]
 * @return {array} Array containing the new ys (same length)
 */

function SavitzkyGolay(ys, xs, options = {}) {
  let { windowSize = 9, derivative = 0, polynomial = 3 } = options;

  if (windowSize % 2 === 0 || windowSize < 5 || !Number.isInteger(windowSize)) {
    throw new RangeError(
      'Invalid window size (should be odd and at least 5 integer number)',
    );
  }
  if (windowSize > ys.length) {
    throw new RangeError(
      `Window size is higher than the data length ${windowSize}>${ys.length}`,
    );
  }
  if (derivative < 0 || !Number.isInteger(derivative)) {
    throw new RangeError('Derivative should be a positive integer');
  }
  if (polynomial < 1 || !Number.isInteger(polynomial)) {
    throw new RangeError('Polynomial should be a positive integer');
  }
  if (polynomial >= 6) {
    // eslint-disable-next-line no-console
    console.warn(
      'You should not use polynomial grade higher than 5 if you are' +
        ' not sure that your data arises from such a model. Possible polynomial oscillation problems',
    );
  }

  let half = Math.floor(windowSize / 2);
  let np = ys.length;
  let ans = new Array(np);
  let weights = fullWeights(windowSize, polynomial, derivative);
  let hs = 0;
  let constantH = true;
  if (Array.isArray(xs)) {
    constantH = false;
  } else {
    hs = Math.pow(xs, derivative);
  }

  //For the borders
  for (let i = 0; i < half; i++) {
    let wg1 = weights[half - i - 1];
    let wg2 = weights[half + i + 1];
    let d1 = 0;
    let d2 = 0;
    for (let l = 0; l < windowSize; l++) {
      d1 += wg1[l] * ys[l];
      d2 += wg2[l] * ys[np - windowSize + l];
    }
    if (constantH) {
      ans[half - i - 1] = d1 / hs;
      ans[np - half + i] = d2 / hs;
    } else {
      hs = getHs(xs, half - i - 1, half, derivative);
      ans[half - i - 1] = d1 / hs;
      hs = getHs(xs, np - half + i, half, derivative);
      ans[np - half + i] = d2 / hs;
    }
  }

  //For the internal points
  let wg = weights[half];
  for (let i = windowSize; i <= np; i++) {
    let d = 0;
    for (let l = 0; l < windowSize; l++) d += wg[l] * ys[l + i - windowSize];
    if (!constantH) hs = getHs(xs, i - half - 1, half, derivative);
    ans[i - half - 1] = d / hs;
  }
  return ans;
}

function getHs(h, center, half, derivative) {
  let hs = 0;
  let count = 0;
  for (let i = center - half; i < center + half; i++) {
    if (i >= 0 && i < h.length - 1) {
      hs += h[i + 1] - h[i];
      count++;
    }
  }
  return Math.pow(hs / count, derivative);
}

function GramPoly(i, m, k, s) {
  let Grampoly = 0;
  if (k > 0) {
    Grampoly =
      ((4 * k - 2) / (k * (2 * m - k + 1))) *
        (i * GramPoly(i, m, k - 1, s) + s * GramPoly(i, m, k - 1, s - 1)) -
      (((k - 1) * (2 * m + k)) / (k * (2 * m - k + 1))) *
        GramPoly(i, m, k - 2, s);
  } else {
    if (k === 0 && s === 0) {
      Grampoly = 1;
    } else {
      Grampoly = 0;
    }
  }
  return Grampoly;
}

function GenFact(a, b) {
  let gf = 1;
  if (a >= b) {
    for (let j = a - b + 1; j <= a; j++) {
      gf *= j;
    }
  }
  return gf;
}

function Weight(i, t, m, n, s) {
  let sum = 0;
  for (let k = 0; k <= n; k++) {
    //console.log(k);
    sum +=
      (2 * k + 1) *
      (GenFact(2 * m, k) / GenFact(2 * m + k + 1, k + 1)) *
      GramPoly(i, m, k, 0) *
      GramPoly(t, m, k, s);
  }
  return sum;
}

/**
 *
 * @param m  Number of points
 * @param n  Polynomial grade
 * @param s  Derivative
 */
function fullWeights(m, n, s) {
  let weights = new Array(m);
  let np = Math.floor(m / 2);
  for (let t = -np; t <= np; t++) {
    weights[t + np] = new Array(m);
    for (let j = -np; j <= np; j++) {
      weights[t + np][j + np] = Weight(j, t, np, n, s);
    }
  }
  return weights;
}

/*function entropy(data,h,options){
    var trend = SavitzkyGolay(data,h,trendOptions);
    var copy = new Array(data.length);
    var sum = 0;
    var max = 0;
    for(var i=0;i<data.length;i++){
        copy[i] = data[i]-trend[i];
    }

    sum/=data.length;
    console.log(sum+" "+max);
    console.log(stat.array.standardDeviation(copy));
    console.log(Math.abs(stat.array.mean(copy))/stat.array.standardDeviation(copy));
    return sum;

}



function guessWindowSize(data, h){
    console.log("entropy "+entropy(data,h,trendOptions));
    return 5;
}
*/

module.exports = SavitzkyGolay;

},{}]},{},[1]);
