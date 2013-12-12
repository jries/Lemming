package org.lemming.processor;

import java.util.ArrayList;
import java.util.List;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.XYFLocalization;
import org.lemming.processor.SISO;

public class PeakFinder extends SISO<Frame,Localization> {

	/** The intensity of a pixel must be greater than {@code threshold} to be considered a local maximum */
	double threshold;

	public PeakFinder(double threshold) {
		this.threshold = threshold;
 	}
	
	@Override
	public void process(Frame frame) {
		//double[] pixels = (double[]) frame.getPixels();
		float[] pixels = (float[]) frame.getPixels();
		
		//for now just print the results to the console
		//List<Integer> localMax = new ArrayList<Integer>();
		
		int width = frame.getWidth();
		long frameNo = frame.getFrameNumber();
		int n = pixels.length - width - 1;
		for (int i = width+1; i < n; i++) {
			if (pixels[i] > threshold) {
				float v = pixels[i];
				if (v >= pixels[i-1-width])
					if (v >= pixels[i-width])
						if (v >= pixels[i+1-width])
							if (v >= pixels[i-1])
								if (v >= pixels[i+1])
									if (v >= pixels[i-1+width])
										if (v >= pixels[i+width])
											if (v >= pixels[i+1+width]){
												output.put(new XYFLocalization(frameNo, i%width, i/width));
												//localMax.add(i);
											}
			}
		}
		
		// TODO we have to keep track of which localization is within which frame
		
		//System.out.println(Long.toString(frameNo)+":"+localMax.toString());
	}
	
}
