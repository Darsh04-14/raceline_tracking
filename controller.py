# controller.py

import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# basic config

dt = 1e-1

# from config.txt
steerKp = 14.2
steerKi = 14.8
steerKd = 0.3

velKp = 6.0
velKi = 3.0
velKd = 0.3

lookaheadBase = 8.0
lookaheadGain = 0.30
lookaheadMax = 30.0

maxSpeedFraction = 1.0
minSpeed = 3.0
curvSpeedGain = 35.0
cteSpeedGain = 0.8

speedTrapCooldown = 50

straightLookaheadM = 80.0
straightCurvThresh = 1e-3
straightCteThresh = 1.0

steerInt = 0.0
velInt = 0.0
prevSteerErr = 0.0
prevVelErr = 0.0
pidFirstRun = True


def resetPidState() -> None:
    global steerInt, velInt, prevSteerErr, prevVelErr, pidFirstRun
    steerInt = 0.0
    velInt = 0.0
    prevSteerErr = 0.0
    prevVelErr = 0.0
    pidFirstRun = True


def wrapAngle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def prepareTrackCache(racetrack: RaceTrack) -> None:
    if hasattr(racetrack, "controllerCache"):
        return

    useRaceline = False

    baseCenterline = getattr(racetrack, "centerline", None)
    raceline = getattr(racetrack, "raceline", None)

    centerline = None

    if raceline is not None:
        arr = np.asarray(raceline)
        if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 2:
            centerline = arr[:, :2]
            useRaceline = True

    if centerline is None:
        if baseCenterline is None:
            centerline = np.zeros((0, 2), dtype=float)
        else:
            centerline = np.asarray(baseCenterline)
            if centerline.ndim != 2 or centerline.shape[1] < 2:
                centerline = np.zeros((0, 2), dtype=float)
            else:
                centerline = centerline[:, :2]

    pointCount = centerline.shape[0]

    if pointCount == 0:
        racetrack.controllerCache = {
            "centerline": centerline,
            "headings": np.zeros(0),
            "signed_curvature": np.zeros(0),
            "avg_seg_length": 1.0,
            "use_raceline": useRaceline,
        }
        return

    segments = np.diff(centerline, axis=0)
    segLengths = np.linalg.norm(segments, axis=1)
    segLengths[segLengths < 1e-6] = 1e-6

    segHeadings = np.arctan2(segments[:, 1], segments[:, 0])

    pointHeadings = np.empty(pointCount, dtype=float)
    pointHeadings[:-1] = segHeadings
    pointHeadings[-1] = segHeadings[-1]

    signedCurvature = np.zeros(pointCount, dtype=float)
    for i in range(1, pointCount):
        dtheta = wrapAngle(pointHeadings[i] - pointHeadings[i - 1])
        signedCurvature[i] = dtheta / segLengths[i - 1]
    if pointCount > 1:
        signedCurvature[0] = signedCurvature[1]

    avgSegLength = float(np.mean(segLengths)) if pointCount > 1 else 1.0

    racetrack.controllerCache = {
        "centerline": centerline,
        "headings": pointHeadings,
        "signed_curvature": signedCurvature,
        "avg_seg_length": avgSegLength,
        "use_raceline": useRaceline,
    }


def closestPointOnCenterline(position: np.ndarray, racetrack: RaceTrack) -> int:
    cache = racetrack.controllerCache
    centerline = cache["centerline"]
    pointCount = centerline.shape[0]

    if pointCount == 0:
        return 0

    lastIndex = cache.get("last_index", None)

    if lastIndex is None:
        dists = np.linalg.norm(centerline - position[None, :], axis=1)
        idx = int(np.argmin(dists))
    else:
        window = 25
        idxRange = np.arange(lastIndex - window, lastIndex + window + 1)
        idxRangeMod = idxRange % pointCount
        candidates = centerline[idxRangeMod]
        dists = np.linalg.norm(candidates - position[None, :], axis=1)
        localBest = int(np.argmin(dists))
        idx = int(idxRangeMod[localBest])

    cache["last_index"] = idx
    return idx


def distanceToCenterline(position: np.ndarray, centerline: np.ndarray) -> float:
    if centerline.shape[0] == 0:
        return 0.0
    dists = np.linalg.norm(centerline - position[None, :], axis=1)
    return float(np.min(dists))


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    global steerInt, velInt, prevSteerErr, prevVelErr, pidFirstRun

    desired = np.asarray(desired, dtype=float)
    state = np.asarray(state, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    assert desired.shape == (2,)

    delta = state[2]
    v = state[3]

    deltaMin, deltaMax = parameters[1], parameters[4]
    vMin, vMax = parameters[2], parameters[5]
    steerRateMin, steerRateMax = parameters[7], parameters[9]
    accelMin, accelMax = parameters[8], parameters[10]

    deltaDes = float(np.clip(desired[0], deltaMin, deltaMax))
    vDes = float(np.clip(desired[1], vMin, vMax))

    steerErr = wrapAngle(deltaDes - delta)

    if pidFirstRun:
        prevSteerErr = steerErr
        prevVelErr = vDes - v
        pidFirstRun = False

    steerInt += steerErr * dt
    steerInt = float(np.clip(steerInt, -0.5, 0.5))

    steerDer = (steerErr - prevSteerErr) / dt
    prevSteerErr = steerErr

    steerRateCmd = (
        steerKp * steerErr
        + steerKi * steerInt
        + steerKd * steerDer
    )

    steerRateCmd = float(np.clip(steerRateCmd, steerRateMin, steerRateMax))

    velErr = vDes - v

    velInt += velErr * dt
    velInt = float(np.clip(velInt, -10.0, 10.0))

    velDer = (velErr - prevVelErr) / dt
    prevVelErr = velErr

    accelCmd = (
        velKp * velErr
        + velKi * velInt
        + velKd * velDer
    )

    accelCmd = float(np.clip(accelCmd, accelMin, accelMax))

    return np.array([steerRateCmd, accelCmd])


prevSpeedTrapAhead = False
lookaheadWarmUp = 10


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    global lookaheadWarmUp, prevSpeedTrapAhead

    state = np.asarray(state, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    prepareTrackCache(racetrack)
    cache = racetrack.controllerCache

    centerline = cache["centerline"]
    headings = cache["headings"]
    signedCurvature = cache["signed_curvature"]
    avgSegLength = cache["avg_seg_length"]

    if centerline.shape[0] == 0:
        return np.array([0.0, minSpeed], dtype=float)

    position = state[0:2]
    v = state[3]
    heading = state[4]

    vMin, vMax = parameters[2], parameters[5]
    deltaMin, deltaMax = parameters[1], parameters[4]
    wheelbase = parameters[0]

    idx = closestPointOnCenterline(position, racetrack)
    pointCount = centerline.shape[0]

    localCurv = float(abs(signedCurvature[idx]))

    lookaheadDistance = lookaheadBase + \
        lookaheadWarmUp + lookaheadGain * max(v, 0.0)
    lookaheadWarmUp = max(lookaheadWarmUp - 0.1, 0.0)

    highSpeed = v > 60.0
    sharpCorner = localCurv > 0.004
    if highSpeed and sharpCorner:
        lookaheadDistance *= 0.9
    elif highSpeed:
        lookaheadDistance *= 1.5

    lookaheadDistance = float(
        np.clip(lookaheadDistance, lookaheadBase, lookaheadMax))

    offsetPts = max(1, int(lookaheadDistance / max(avgSegLength, 1e-3)))
    targetIdx = (idx + offsetPts) % pointCount
    targetPoint = centerline[targetIdx]

    toTarget = targetPoint - position
    targetHeading = np.arctan2(toTarget[1], toTarget[0])

    alpha = wrapAngle(targetHeading - heading)
    deltaDes = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookaheadDistance)

    if highSpeed:
        softDeltaMax = 0.75 * deltaMax
        softDeltaMin = 0.75 * deltaMin
        deltaDes = float(np.clip(deltaDes, softDeltaMin, softDeltaMax))

    deltaDes = float(np.clip(deltaDes, deltaMin, deltaMax))

    prevIdx = (idx - 1) % pointCount
    nextIdx = (idx + 1) % pointCount
    segment = centerline[nextIdx] - centerline[prevIdx]
    segLen = float(np.linalg.norm(segment))
    if segLen < 1e-6:
        crossTrackErr = 0.0
    else:
        segmentUnit = segment / segLen
        rel = position - centerline[idx]
        crossTrackErr = float(
            segmentUnit[0] * rel[1] - segmentUnit[1] * rel[0]
        )

    shortWindowM = 80.0
    stepsAheadShort = max(5, int(shortWindowM / max(avgSegLength, 1e-3)))

    maxCurvShort = 0.0
    for k in range(stepsAheadShort):
        j = (idx + k) % pointCount
        c = abs(signedCurvature[j])
        if c > maxCurvShort:
            maxCurvShort = c

    effectiveCurvBase = max(localCurv, maxCurvShort)

    speedtrapWindowM = 250.0
    stepsSpeedtrap = max(5, int(speedtrapWindowM / max(avgSegLength, 1e-3)))

    maxCurvSpeedtrap = 0.0
    for k in range(stepsSpeedtrap):
        j = (idx + k) % pointCount
        c = abs(signedCurvature[j])
        if c > maxCurvSpeedtrap:
            maxCurvSpeedtrap = c

    kappaSpeedtrap = 0.042 - ((v / 3.375e6) ** 0.5)

    speedtrapAhead = maxCurvSpeedtrap > kappaSpeedtrap

    vStraightBase = vMax * maxSpeedFraction

    turnLookaheadGain = 2.0

    if speedtrapAhead:
        effectiveCurv = effectiveCurvBase * turnLookaheadGain
    else:
        effectiveCurv = effectiveCurvBase
        speedtrapAhead = False

    if not speedtrapAhead:
        straightBoost = 4.1
        curvGainStraight = 0.5 * curvSpeedGain
        vStraight = vStraightBase * straightBoost
        vLimitCurv = vStraight / (1.0 + curvGainStraight * effectiveCurv)
    else:
        curvGainTrap = 0.27 * curvSpeedGain
        vStraight = 0.5 * vStraightBase
        vLimitCurv = vStraight / (1.0 + curvGainTrap * effectiveCurv)

    vLimitCte = vStraight / (1.0 + cteSpeedGain * abs(crossTrackErr))

    distFromLine = distanceToCenterline(position, centerline)
    safeDistSoft = 3
    safeDistHard = 4.5

    if distFromLine > safeDistHard:
        safetyScale = 0.05
    elif distFromLine > safeDistSoft:
        t = (distFromLine - safeDistSoft) / (safeDistHard - safeDistSoft)
        safetyScale = 0.3 - t * (0.3 - 0.05)
    else:
        safetyScale = 1.0

    if v > 30.0 and distFromLine > 1.5:
        safetyScale *= 0.4

    vTarget = min(vLimitCurv, vLimitCte)

    if lookaheadWarmUp == 0.0:
        vTarget *= safetyScale

    vDes = float(np.clip(vTarget, 0.0, vStraight))

    if vDes < minSpeed and safetyScale > 0.2:
        vDes = minSpeed * safetyScale

    return np.array([deltaDes, vDes])


resetPidState()
