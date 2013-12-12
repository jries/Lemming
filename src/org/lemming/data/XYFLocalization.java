package org.lemming.data;

public class XYFLocalization extends XYLocalization {

	long frame = 0;
	
	public long getFrame() {
		return frame;
	}
	
	public XYFLocalization(long frame, double x, double y) {
		super(x, y);

		this.frame = frame;
	}

	public XYFLocalization(long frame, double x, double y, long ID) {
		super(x, y, ID);

		this.frame = frame;
	}

}
