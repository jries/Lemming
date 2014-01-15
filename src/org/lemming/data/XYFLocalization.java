package org.lemming.data;

/**
 * A localization with X, Y, ID and frame.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public class XYFLocalization extends XYLocalization {

	long frame = 0;

	/**
	 * Get the frame number corresponding from this localization.
	 * 
	 * @return
	 */
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
