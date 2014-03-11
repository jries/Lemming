package org.lemming.inputs;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

import java.util.Random;

import javax.swing.JOptionPane;

import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;
import matlabcontrol.extensions.MatlabTypeConverter;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Localization;
import org.lemming.data.XYLocalization;



/**
 * Open a matlab session and a specified matlab program. The process is stopped until the user is done and then the class import a tiff stack (example of testSTORM).
 * 
 * @author Joe Borbely, Thomas Pengo, Joran Deschamps
 */

public class MatlabSOCommunicator<T extends RealType<T> & NativeType<T>> extends SO<ImgLib2Frame<T>> { 
	
	String path;									// Local path to the directory of the matlab program
	String matfile_name;							// Name of the matlab file .m (should be a command)
	String output_path;								// Path of the output images 	

	MatlabProxyFactoryOptions options;				// Proxy constructor options 
	MatlabProxyFactory factory;						// Proxy factory
	MatlabProxy proxy;								// Proxy in charge of the communications with Matlab
    MatlabTypeConverter processor;					// Converter for variables between Matlab and Java
    
	int curSlice=0;									// Counter for the localizations
	int num_frames;  								// Number of frames created
	
	Boolean user_done = false;
	
	ImagePlus img;
	
	public MatlabSOCommunicator(String s1, String s2, String s3) throws MatlabConnectionException, MatlabInvocationException{

		path = s1;
		matfile_name = s2;
		output_path = s3;
		
		options = new MatlabProxyFactoryOptions.Builder()
        //.setHidden(b)			// Hide the main Matlab window
        .setUsePreviouslyControlledSession(true)
        .build();	
		
		factory = new MatlabProxyFactory(options);
		proxy = factory.getProxy();												// Create proxy between JVM and Matlab
	    processor = new MatlabTypeConverter(proxy);
		
	    proxy.eval("addpath "+path);											
	}

	public void beforeRun(){
		try {
			this.runMatfile();												// Launch Matlab program
		} catch (MatlabInvocationException e) {
			e.printStackTrace();
		}
		
		// Wait until user is done
		JOptionPane.showMessageDialog ( 									// Dialog to pause the program 
				null, "Waiting for end of MATLAB task" );

		user_done = true;

		// Get number of frames produced
		try {
			num_frames = (int)((double[]) proxy.getVariable("seq_params.aq.frame_N"))[0];
		} catch (MatlabInvocationException e) {
			e.printStackTrace();
		}

		// Get frames
		img = new ImagePlus(output_path);

	}
		
	public void afterRun(){
		proxy.disconnect();
	}
	
	void runMatfile() throws MatlabInvocationException{
		proxy.eval(matfile_name);
	}

	public boolean isUserDone() {
		return user_done;
	}

	@Override
	public boolean hasMoreOutputs() {
		return curSlice < img.getStack().getSize();
	}

	@Override
	protected ImgLib2Frame<T> newOutput() {
		curSlice++;
		//System.out.println(curSlice);

		ImageProcessor ip = img.getStack().getProcessor(curSlice);
		
		long[] dims = new long[]{ip.getWidth(), ip.getHeight()};
		
		Img theImage = null;
		if (ip instanceof ShortProcessor) {
			theImage = ArrayImgs.unsignedShorts((short[]) ip.getPixels(), dims);
		} else if (ip instanceof FloatProcessor) {
			theImage = ArrayImgs.floats((float[])ip.getPixels(), dims);
		} else if (ip instanceof ByteProcessor) {
			theImage = ArrayImgs.unsignedBytes((byte[])ip.getPixels(), dims);
		}
		
		return new ImgLib2Frame(curSlice, (int)dims[0], (int)dims[1], theImage);		
	}

	public void show(){
		img.show();
	}

	public int getNumFrames() {
		return num_frames;
	}

}

