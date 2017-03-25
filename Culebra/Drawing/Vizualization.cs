﻿using System;
using System.Collections.Generic;
using Rhino.Geometry;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using System.Drawing;
using System.Reflection;
using ikvm;
using processing.core;
using culebra.behaviors;
using CulebraData;
using CulebraData.Objects;
using CulebraData.Utilities;

namespace CulebraData.Drawing
{
    /// <summary>
    /// Visualization Class - Used to access Creeper Object's Viz properties
    /// </summary>
    public class Vizualization
    {
        /// <summary>
        /// Draws a Point Graphic through the display pipeline
        /// </summary>
        /// <param name="args">preview Display Args for IGH_PreviewObjects</param>
        /// <param name="particleList">the list of points representing the particle positions</param>
        public void DrawPointGraphic(IGH_PreviewArgs args, List<Point3d> particleList)
        {
            foreach (Point3d p in particleList)
                args.Display.DrawPoint(p, System.Drawing.Color.Blue);
        }
        /// <summary>
        /// Draws a Particle through the display pipeline
        /// </summary>
        /// <param name="args">preview Display Args for IGH_PreviewObjects</param>
        /// <param name="file">the texture file</param>
        /// <param name="particleSystem">the particle system</param>
        public void DrawParticles(IGH_PreviewArgs args, string file, ParticleSystem particleSystem)
        {
            Bitmap bm = new Bitmap(file);
            Rhino.Display.DisplayBitmap dbm = new Rhino.Display.DisplayBitmap(bm);
            args.Display.DrawParticles(particleSystem, dbm);
        }
        /// <summary>
        /// Draws Sprites through the display pipeline
        /// </summary>
        /// <param name="args">preview Display Args for IGH_PreviewObjects</param>
        /// <param name="file">the texture file</param>
        /// <param name="particleList">the list of points representing the particle positions</param>
        public void DrawSprites(IGH_PreviewArgs args, string file, List<Point3d> particleList)
        {
            Bitmap bm = new Bitmap(file);
            Rhino.Display.DisplayBitmap dbm = new Rhino.Display.DisplayBitmap(bm);
            Rhino.Display.DisplayBitmapDrawList ddl = new Rhino.Display.DisplayBitmapDrawList();         
            ddl.SetPoints(particleList, Color.FromArgb(25, 255, 255, 255));
            args.Display.DrawSprites(dbm, ddl, 2.0f, new Vector3d(0, 0, 1), true);
        }
        /// <summary>
        /// Draws a gradient trail through the display pipeline
        /// </summary>
        /// <param name="args">preview Display Args for IGH_PreviewObjects</param>
        /// <param name="file"></param>
        /// <param name="particleSet">The data tree containing the points list for each object you want to draw a gradient for</param>
        /// <param name="colorType">the color type</param>
        /// <param name="minTrailThickness">the minimum trail thickness</param>
        /// <param name="maxTrailThickness">the maximum trail thickness</param>
        public void DrawGradientTrails(IGH_PreviewArgs args, string file, DataTree<Point3d> particleSet, int colorType, float minTrailThickness, float maxTrailThickness)
        {
            Color color = args.WireColour;
            for (int i = 0; i < particleSet.BranchCount; i++)
            {
                List<Point3d> ptlist = particleSet.Branch(i);
                //-------DRAW TRAILS AS SEGMENTS WITH CUSTOM STROKE WIDTH---------
                if (ptlist.Count > 0)
                {
                    for (int x = 0; x < ptlist.Count; x++)
                    {
                        if (x != 0)
                        {
                            float stroke = CulebraData.Utilities.Convert.Map(x / (1.0f * ptlist.Count), 0.0f, 1.0f, minTrailThickness, maxTrailThickness);
                            float colorValue = CulebraData.Utilities.Convert.Map(x / (1.0f * ptlist.Count), 0.0f, 1.0f, 0f, 255.0f);
                            if(colorType == 0)
                            {
                                args.Display.DrawLine(ptlist[x - 1], ptlist[x], Color.FromArgb(0, (int)colorValue, 0, 100), (int)stroke);
                            }
                            else if(colorType == 1)
                            {
                                args.Display.DrawLine(ptlist[x - 1], ptlist[x], Color.FromArgb(0, 0, 255, (int)colorValue), (int)stroke);
                            }
                            else
                            {
                                args.Display.DrawLine(ptlist[x - 1], ptlist[x], Color.FromArgb(0, 255, 255, (int)colorValue), (int)stroke);
                            }
                        }
                    }
                }
            }
        }
        /// <summary>
        /// Draws a polyline trail through the display pipeline
        /// </summary>
        /// <param name="args">preview Display Args for IGH_PreviewObjects</param>
        /// <param name="particleSet">The data tree containing the points list for each object you want to draw a gradient for</param>
        /// <param name="dottedPolyline">do you want a dotted polyline</param>
        /// <param name="thickness">the thickness of the trail</param>
        public void DrawPolylineTrails(IGH_PreviewArgs args, DataTree<Point3d> particleSet, bool dottedPolyline, int thickness)
        {
            Color color = args.WireColour;
            for (int i = 0; i < particleSet.BranchCount; i++)
            {
                List<Point3d> ptlist = particleSet.Branch(i);
                if (dottedPolyline)
                {
                    args.Display.DrawDottedPolyline(ptlist, Color.FromArgb(0, 255, 0, 255), false);
                }
                else
                {
                    args.Display.DrawPolyline(ptlist, Color.FromArgb(0, 255, 0, 255), thickness);
                }
            }
        }
        /// <summary>
        /// Draws a disco trail through the display pipeline. Trails flash different colors throughout the simulation
        /// </summary>
        /// <param name="args">preview Display Args for IGH_PreviewObjects</param>
        /// <param name="file">the texture file</param>
        /// <param name="particleSet">The data tree containing the points list for each object you want to draw a gradient for</param>
        /// <param name="randomGen">an instance of the random class</param>
        /// <param name="minTrailThickness">the minimum trail thickness</param>
        /// <param name="maxTrailThickness">the maximum trail thickness</param>
        public void DrawDiscoTrails(IGH_PreviewArgs args, string file, DataTree<Point3d> particleSet, Random randomGen, float minTrailThickness, float maxTrailThickness)
        {
            Color color = args.WireColour;
            for (int i = 0; i < particleSet.BranchCount; i++)
            {
                List<Point3d> ptlist = particleSet.Branch(i);
                //-------DRAW TRAILS AS SEGMENTS WITH CUSTOM STROKE WIDTH---------
                Color randomColorAction = CulebraData.Utilities.Convert.GetRandomColor(randomGen);
                if (ptlist.Count > 0)
                {
                    for (int x = 0; x < ptlist.Count; x++)
                    {
                        if (x != 0)
                        {
                            float stroke = CulebraData.Utilities.Convert.Map(x / (1.0f * ptlist.Count), 0.0f, 1.0f, minTrailThickness, maxTrailThickness);
                            args.Display.DrawLine(ptlist[x - 1], ptlist[x], randomColorAction, (int)stroke);
                        }
                    }
                }
            }
        }
    }
}
