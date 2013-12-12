package org.lemming.data;

import java.util.Arrays;

public class XYFwLocalization extends XYFLocalization {

	float[] window;
	
	public float[] getWindow() { return window; }

	public XYFwLocalization(float[] window, long frame, double x, double y) {
		super(frame, x, y);
		
		this.window = Arrays.copyOf(window, window.length);
	}
	
	public XYFwLocalization(float[] window, long frame, double x, double y, long ID) {
		super(frame, x, y, ID);
		
		this.window = Arrays.copyOf(window, window.length);
	}
	
}
