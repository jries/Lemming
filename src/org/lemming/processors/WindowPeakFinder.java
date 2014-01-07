package org.lemming.processors;

import org.lemming.data.Frame;
import org.lemming.data.XYFLocalization;
import org.lemming.data.XYFwLocalization;

public class WindowPeakFinder extends PeakFinder {

	public WindowPeakFinder(double threshold) {
		super(threshold);
	}
	
	int size = 1;	

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
												float [] w = getWindow(pixels, width, i%width, i/width, size);
												
												output.put(new XYFwLocalization(w, frameNo, i%width, i/width));
												//localMax.add(i);
											}
			}
		}
		
		// TODO we have to keep track of which localization is within which frame
		
		//System.out.println(Long.toString(frameNo)+":"+localMax.toString());
	}

	private float[] getWindow(float[] pixels, int width, int i, int j, int siz) {
		float[] win = new float[ (2*siz+1)*(2*siz+1) ];
		
		for (int di = -siz; di<=siz; di++)
			for (int dj = -siz; dj<=siz; dj++)
				win[ (dj+siz)*(2*siz+1)+ di+siz ] = pixels[ j*width + i ];
		
		return win;
	}
}
