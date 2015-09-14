package org.lemming.modules;

import com.amd.aparapi.Kernel;

import ij.ImagePlus;
import ij.process.FloatProcessor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;
import org.lemming.interfaces.Element;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.MultiRunModule;
import org.lemming.tools.ArrayTools;

import java.util.Random;

public class DummyImageLoader extends MultiRunModule{
	
	private int curSlice = 0;
	private final int particlesPerFrame, stackSize, width, height;
	private long start;
	final Random random = new Random();

	static Kernel_subPixelGaussianRendering myAddParticles = new Kernel_subPixelGaussianRendering();


	public DummyImageLoader(int particlesPerFrame, int stackSize, int width, int height) {
		this.stackSize = stackSize;
		this.width = width;
		this.height = height;
		this.particlesPerFrame = particlesPerFrame;
	}
	
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		iterator = outputs.keySet().iterator().next();
	}

	@Override
	public Element process(Element data) {
		

		float[] intensity = new float[particlesPerFrame];
		float[] sigmaX = new float[particlesPerFrame];
		float[] sigmaY = new float[particlesPerFrame];
		float[] x = new float[particlesPerFrame];
		float[] y = new float[particlesPerFrame];

		FloatProcessor fp = new FloatProcessor(width, height);
		for (int n=0; n<particlesPerFrame; n++) {
			intensity[n] = 1000;
			sigmaX[n] = 1.5f;
			sigmaY[n] = 1.5f;
			x[n] = random.nextFloat()*(width-1);
			y[n] = random.nextFloat()*(height-1);
		}
		fp = myAddParticles.drawParticles(fp, intensity, sigmaX, sigmaY, x, y, 10);
		Img<FloatType> img = ArrayImgs.floats((float[])fp.getPixels(), new long[]{width, height});
		curSlice++;

		ImgLib2Frame<FloatType> frame = new ImgLib2Frame<>(curSlice, width, height, img);
		if (curSlice >= stackSize){
			frame.setLast(true);
			cancel();
		}
		new ImagePlus("",fp).show();
		return frame;
	}
	
	@Override
	public void afterRun(){
		System.out.println("Loading done in " + (System.currentTimeMillis()-start) + "ms.");
	}
	

	@Override
	public boolean check() {
		return outputs.size()>=1;
	}
}

class Kernel_subPixelGaussianRendering extends Kernel {
	private float [] pixels;
	private int [] pixels_encodedFloatToInt;
	private float [] intensity_$constant$;
	private float [] sigmaX_$constant$;
	private float [] sigmaY_$constant$;
	private float [] x_$constant$;
	private float [] y_$constant$;
	private int width, height, subPixels;
	public int precision = 3; // decimal places
	private int precisionMultiplier = (int) pow(10, precision);

	public FloatProcessor drawParticles(FloatProcessor fp,
										float[] intensity, float[] sigmaX, float[] sigmaY, float[] x, float[] y, int subPixels){


		this.pixels = (float[]) fp.getPixels();
		this.pixels_encodedFloatToInt = ArrayTools.encodeFloatArrayIntoInt(pixels, precision);
		this.width = fp.getWidth();
		this.height = fp.getHeight();
		this.subPixels = subPixels;
		this.intensity_$constant$ = intensity;
		this.sigmaX_$constant$ = sigmaX;
		this.sigmaY_$constant$ = sigmaY;
		this.x_$constant$ = x;
		this.y_$constant$ = y;

		setExplicit(true);
		put(this.pixels_encodedFloatToInt);
		put(this.intensity_$constant$);
		put(this.sigmaX_$constant$);
		put(this.sigmaY_$constant$);
		put(this.x_$constant$);
		put(this.y_$constant$);
		execute(this.intensity_$constant$.length);
		get(this.pixels_encodedFloatToInt);

		return new FloatProcessor(width, height, this.pixels);
	}

	@Override
	public void run() {
		int p = getGlobalId(0);
		float x = x_$constant$[p] * subPixels;
		float y = y_$constant$[p] * subPixels;
		int rx = round(x);
		int ry = round(y);
		float intensity = intensity_$constant$[p] / subPixels / subPixels;
		float sigmaX = sigmaX_$constant$[p] * subPixels;
		float sigmaY = sigmaY_$constant$[p] * subPixels;
		float sigmaX2 = 2*pow(sigmaX, 2);
		float sigmaY2 = 2*pow(sigmaY, 2);
		float v;

		int radiusX = (int) (sigmaX * 2.354f) + 3;
		int radiusY = (int) (sigmaY * 2.354f) + 3;

		int i_;
		int j_;

		for (int i = rx-radiusX; i<=rx+radiusX; i++) {
			i_ = i / subPixels;
			if (i_ < 0 || i_ > width - 1) continue;

			for (int j = ry-radiusY; j<=ry+radiusY; j++) {
				j_ = j / subPixels;
				if (j_ < 0 || j_ > height - 1) continue;

				v = intensity * exp(-pow((i+0.5f-x),2)/sigmaX2-pow((j+0.5f-y), 2)/sigmaY2);
				atomicAdd(pixels_encodedFloatToInt, j_ * width + i_, round(v* precisionMultiplier));
			}
		}
	}
}