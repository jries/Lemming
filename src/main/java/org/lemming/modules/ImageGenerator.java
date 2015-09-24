package org.lemming.modules;

// some functions are adapted from thunderSTORM

import ij.ImagePlus;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;
import ij.process.ShortProcessor;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.NumericType;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.lemming.interfaces.Element;
import org.lemming.math.Gaussian;
import org.lemming.pipeline.ImgLib2Frame;
import org.lemming.pipeline.MultiRunModule;
import org.lemming.tools.LemmingUtils;

public class ImageGenerator<T extends NumericType<T> & NativeType<T>> extends MultiRunModule {

	private int curSlice = 0;
	private int width, height;
	private int numFrames;
	private long start;
	private ImagePlus img;
	private int numMol;
	private int boxSize;
	private int I;
	private double sigma;
	private int Bg;
			
    private RandomDataGenerator rand; 
    
    public ImageGenerator(int width, int height, int numFrames, int numMol, int I, double sigma, int Bg, int boxSize){
    	this.width = width;
    	this.height = height;
    	this.numFrames = numFrames;
    	this.numMol = numMol;
    	this.boxSize = boxSize;
    	this.I = I;
    	this.sigma = sigma;
    	this.Bg = Bg;
		RandomDataGenerator rand = new RandomDataGenerator();
    }
		
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
		iterator = outputs.keySet().iterator().next();
	}

	@Override
	public void afterRun(){
		System.out.println("Generating done in " + (System.currentTimeMillis()-start) + "ms.");
	}
	
	public void show(){
		img.show();
	}
	
	@Override
	public Element process(Element data) {
		if (curSlice >= numFrames){ cancel(); return null; }
		
		ImageProcessor ip = generateFrame(numMol, I, sigma, Bg);
		
		if(curSlice == 0){
			img = new ImagePlus("Generated image",ip);
		} else {
			img.getStack().addSlice(ip);
		}
		curSlice++;
		
		Img<T> theImage = LemmingUtils.wrap(ip);
		
		ImgLib2Frame<T> frame = new ImgLib2Frame<>(curSlice, ip.getWidth(), ip.getHeight(), theImage);
		if (curSlice >= numFrames)
			frame.setLast(true);
		
		return frame;
	}
	
	@Override
	public boolean check() {

		return false;
	}
	
	///////////////////////////////////////////////////////////////////////
	// Generator
	public ShortProcessor generateFrame(int N, int Nphotons, double sigma, int background){
		rand = new RandomDataGenerator();
		
        short [] zeros = new short[width*height];
        ShortProcessor frame = new ShortProcessor(width, height, zeros, null);
        
        frame = addBackground(frame,background);
        
        Gaussian psf = new Gaussian();
        double[] params = {0,0,sigma,Nphotons,0};				// here units in photons but might have to put in camera signal...
        
        for(int i = 0;i<N;i++) {
        	params[0] = rand.nextInt(0,width);
        	params[1] = rand.nextInt(0,height);

        	frame = addMolecule(frame,psf,params,width,height);
        }
        
        //simulates poisson distributed photon arrival 
        frame = samplePoisson(frame);

        // Camera noise and gain
        
        //convert to integer
        return (ShortProcessor)frame.convertToShort(false);
	}
	
    @SuppressWarnings("static-access")
	private ShortProcessor addMolecule(ShortProcessor fp, Gaussian psf, double[] params, int width, int height){
    	for(int i=(int) (params[0]-boxSize); i<params[0]+boxSize;i++){
    		for(int j=(int) (params[1]-boxSize); j<params[1]+boxSize;j++){
    			if(!isOutsideBounds(i,j,width,height)){
    				fp.set(i, j, fp.get(i, j) + (int) psf.getValue(params, i, j));
    			}
    		}
    	}
    	return fp;
    }
	
    private boolean isOutsideBounds(int x, int y, int width, int height){
    	if(x<0 || x>=width || y<0 || y>=height)
    		return true;
    	
    	return false;
    }
    /**
     * Replaces each pixel value with a sample from a poisson distribution with mean value equal to the pixel original value.
     */
    private ShortProcessor samplePoisson(ShortProcessor fp){
        for(int i = 0; i < fp.getPixelCount(); i ++){
            float mean =  fp.getf(i);

            double value = mean > 0 ? (rand.nextPoisson(mean)) : 0;
            fp.setf(i, (float)value);
        }
        return fp;
    }
    
    /**
     * Replaces each pixel value with a sample from a Gamma distribution with shape equal to the original pixel value and scale equal to the gain parameter.
     */
    private ShortProcessor sampleGamma(ShortProcessor fp, double gain){
        for(int i = 0; i < fp.getPixelCount(); i ++){
            double value = fp.getf(i);
            value = rand.nextGamma(value + 1e-10, gain);
            fp.setf(i, (float)value);
        }
        return fp;
    }

    private ShortProcessor addBackground(ShortProcessor fp, int bg){
        for(int i = 0; i < fp.getPixelCount(); i ++){
            fp.setf(i, fp.get(i)+(short)rand.nextInt((int) (0.1*bg), (int) (1.9*bg)));
        }
        return fp;
    }
    
    private void testImage(ImageProcessor ip){
    	ImageStatistics is = ip.getStatistics();
    	System.out.println("Mean : "+is.mean);
    	System.out.println("Max : "+is.max);
    }
}
