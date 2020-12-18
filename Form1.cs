using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace AOCI_6
{
    public partial class Form1 : Form
    {
        int frameCounter = 0;
        private MotionDetector resultImage = new MotionDetector();

        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            resultImage.ImageProcessed += ImageProcessed;
            resultImage.StartVideoFromCam();
        }
        private void ImageProcessed(object sender, MotionDetector.ImageEventArgs e)
        {
            imageBox2.Image = e.Image;
            imageBox1.Image = e.ImageOriginal;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            imageBox2.Image = resultImage.timerVideo();
            imageBox1.Image = resultImage.timerVideoOriginal();

            if (frameCounter >= resultImage.capture.GetCaptureProperty(CapProp.FrameCount))
                timer1.Enabled = false;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            var result = openFileDialog.ShowDialog();
            if (result == DialogResult.OK)
            {
                string fileName = openFileDialog.FileName;

                resultImage.VideoProcessing(fileName);

                //var frameRate = resultImage.capture.GetCaptureProperty(CapProp.Fps);
                //timer1.Interval = 1 / (int)frameRate;
                timer1.Interval = 25;

                timer1.Enabled = true;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var frame = new Mat();
            resultImage.capture.Retrieve(frame);

            resultImage.bg = frame.ToImage<Gray, byte>();
        }
    }
}
