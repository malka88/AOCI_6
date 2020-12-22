using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace AOCI_6
{
    class MotionDetector
    {
        public Image<Bgr, byte> sourceImage;
        public event EventHandler<ImageEventArgs> ImageProcessed;
        public VideoCapture capture;

        public Image<Gray, byte> bg { get; set; } = null;
        int frameCounter = 0;
        int frameCounterOriginal = 0;

        BackgroundSubtractorMOG2 subtractor = new BackgroundSubtractorMOG2(1000, 32, true);

        public class ImageEventArgs : EventArgs
        {
            public IInputArray Image { get; set; }
            public IInputArray ImageOriginal { get; set; }
        }    

        public void StartVideoFromCam()
        {
            capture = new VideoCapture();
            capture.ImageGrabbed += ProcessFrame;
            capture.Start();
        }

        public Image<Bgr, byte> timerVideo()
        {
            var frame = capture.QueryFrame();

            sourceImage = frame.ToImage<Bgr, byte>(); //обрабатываемое изображение из функции Processing приравниваем к фрейму
            //var videoImage = Processing(); //на финальное изображение накладываем фильтр, вызывая функцию
            capture.Retrieve(frame);
            Image<Gray, byte> cur = frame.ToImage<Gray, byte>();

            var foregroundMask = cur.CopyBlank();
            subtractor.Apply(cur, foregroundMask);

            foregroundMask._ThresholdBinary(new Gray(253), new Gray(255));

            foregroundMask.Erode(3);
            foregroundMask.Dilate(4);
            FilterMask(foregroundMask);

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(
                     foregroundMask,
                     contours,
                     null,
                     RetrType.External, // получение только внешних контуров
                     ChainApproxMethod.ChainApproxTc89L1);

            var output = frame.ToImage<Bgr, byte>().Copy();

            for (int i = 0; i < contours.Size; i++)
            {
                if (CvInvoke.ContourArea(contours[i], false) > 700) //игнорирование маленьких контуров
                {
                    Rectangle rect = CvInvoke.BoundingRectangle(contours[i]);
                    output.Draw(rect, new Bgr(Color.GreenYellow), 1);
                }
            }

            frameCounter++;
            return output;
        }

        public Image<Bgr, byte> timerVideoOriginal()
        {
            var frame = capture.QueryFrame();

            sourceImage = frame.ToImage<Bgr, byte>(); //обрабатываемое изображение из функции Processing приравниваем к фрейму
            //var videoImage = Processing(); //на финальное изображение накладываем фильтр, вызывая функцию
            capture.Retrieve(frame);

            var output = frame.ToImage<Bgr, byte>().Copy();

            frameCounterOriginal++;

            return output;
        }

        public void VideoProcessing(string fileName)
        {
            capture = new VideoCapture(fileName); //берем кадры из видео
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            try
            {
                var frame = new Mat();
                //var frameOrig = new Mat();
                capture.Retrieve(frame);
                //capture.Retrieve(frameOrig);

                Image<Gray, byte> cur = frame.ToImage<Gray, byte>();
                Image<Gray, byte> curOrig = frame.ToImage<Gray, byte>();
                var outputOrig = frame.ToImage<Bgr, byte>().Copy();

                if (bg != null)
                {
                    Image<Gray, byte> diff = bg.AbsDiff(cur);

                    diff._ThresholdBinary(new Gray(100), new Gray(255));

                    diff.Erode(3);
                    diff.Dilate(4);

                    VectorOfVectorOfPoint contoursOrig = new VectorOfVectorOfPoint();
                    CvInvoke.FindContours(
                             diff,
                             contoursOrig,
                             null,
                             RetrType.External, // получение только внешних контуров
                             ChainApproxMethod.ChainApproxTc89L1);

                    //outputOrig = curOrig.ToImage<Bgr, byte>().Copy();

                    for (int i = 0; i < contoursOrig.Size; i++)
                    {
                        if (CvInvoke.ContourArea(contoursOrig[i], false) > 700) //игнорирование маленьких контуров
                        {
                            Rectangle rectOrig = CvInvoke.BoundingRectangle(contoursOrig[i]);
                            outputOrig.Draw(rectOrig, new Bgr(Color.GreenYellow), 1);
                        }
                    }
                }

                //bg = frame.ToImage<Gray, byte>();

                var foregroundMask = cur.CopyBlank();
                subtractor.Apply(cur, foregroundMask);

                foregroundMask._ThresholdBinary(new Gray(253), new Gray(255));

                foregroundMask.Erode(3);
                foregroundMask.Dilate(4);
                FilterMask(foregroundMask);

                VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
                CvInvoke.FindContours(
                         foregroundMask,
                         contours,
                         null,
                         RetrType.External, // получение только внешних контуров
                         ChainApproxMethod.ChainApproxTc89L1);

                var output = frame.ToImage<Bgr, byte>().Copy();

                for (int i = 0; i < contours.Size; i++)
                {
                    if (CvInvoke.ContourArea(contours[i], false) > 700) //игнорирование маленьких контуров
                    {
                        Rectangle rect = CvInvoke.BoundingRectangle(contours[i]);
                        output.Draw(rect, new Bgr(Color.GreenYellow), 1);
                    }
                }

                ImageProcessed?.Invoke(this, new ImageEventArgs { Image = output, ImageOriginal = outputOrig });
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
                capture.ImageGrabbed -= ProcessFrame;
                capture.Stop();
            }
        }

        private Image<Gray, byte> FilterMask(Image<Gray, byte> mask)
        {
            var anchor = new Point(-1, -1);
            var borderValue = new MCvScalar(1);
            // создание структурного элемента заданного размера и формы для морфологических операций
            var kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(3, 3), anchor);
            // заполнение небольших тёмных областей
            var closing = mask.MorphologyEx(MorphOp.Close, kernel, anchor, 1, BorderType.Default,
            borderValue);
            // удаление шумов
            var opening = closing.MorphologyEx(MorphOp.Open, kernel, anchor, 1, BorderType.Default,
            borderValue);
            // расширение для слияния небольших смежных областей
            var dilation = opening.Dilate(7);
            // пороговое преобразование для удаления теней
            var threshold = dilation.ThresholdBinary(new Gray(240), new Gray(255));
            return threshold;
        }
    }
}
