package org.lemming.pipeline;

import net.imglib2.Interval;

public class Kernel {
	
	final private long ID;
	final private Interval roi;
	final private long frame;
	final private float[] values; 

	public Kernel(long id, long frame, Interval roi, float[] values) {
		this.ID = id;
		this.roi = roi;
		this.frame = frame;
		this.values = values;
	}

	public Interval getRoi() {
		return roi;
	}

	public long getFrame() {
		return frame;
	}

	public float[] getValues() {
		return values;
	}
	
	public long getID(){
		return ID;
	}
	
}
