---
layout: post
title:  "Thoughts on Tesla Supervised FSD"
date:   2024-04-01 9:00:00 -0700
categories: Robotics
usemathjax: true
---

# Overview
Yesterday, Telsa gave me a month long free trial of FSD. Naturally, I immediately went out to try it. Here's some observations and some of my thoughts.

I live in currently live in SF so assume everything I talk about here is within the SF Bay Area.

# Highlights
- Immediately did terribly on market st.
- Wheel was very jerky through turns. Action prediction horizon likely overfit
- False red light near twin peaks. Should’ve been able to infer that signal wasn’t pointed at the car
- Almost swerved into a bus trying to cut around it when it shouldn’t have
- Visualization indicates low resolution with much more map filling up close

# Analysis
Two particular observations stood out to me as salient. 

1. The car approached a weird shaped curb that caught my eye. As it got closer, I noticed that the visualization on the dashboard was still very murky. This is concerning as the car needed to make a left hand turn around the weird shaped curb. Only a few feet away did the curb really get clearly defined in the visualization
2. jerkiness through tough corners

Together, these feel like problems caused by low angular camera resolution. Angular camera resolution is the amount of pixels the camera has for a given field of view. So even if you have a 4k camera, if you slap a fisheye lens on it, it's angular resolution would be shit.

The curb shape being resolved only when the car comes close enough to see it's shape is a clear indication of this. Of course, I don't know the details for the update frequency of the visualizer versus the planning stack so it's entirely possible I'm mis-attributing the crappy turn to the late visualization of the curb.

I believe that the jerkiness around turning is also related to camera resolution. I believe that as the car completes the turn, more information becomes available to it, resulting in the planning stack rapidly trying to fit it's motion to the new information. It's entirely possible this is a compute restriction; higher resolution = higher latency. To do high frequency control with the camera stack may not be possible requiring bifurcating the camera stream to a high resolution pass and a low resolution pass. The high resolution pass runs much less frequently than the low resolution pass which runs on the fastest control loop frequency. So this may not a camera hw problem but a drive compute problem. Maybe solvable through algorithmic wins (small networks that can process more pixels with the same cycle budget)?

# Takeaways
Assuming the camera resolution is the problem, it's my belief that this would be really difficult to solve. In my opinion, the only thing to do keeping vision only is to switch to foveated cameras that are able to actually swivel. This could keep costs low while dramatically increasing the effective angular resolution by trading camera latency at the periphery. Or you can just wait till there's better cameras :).