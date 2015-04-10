package org.lemming.data;

/**
 * A localization with X, Y, ID and frame.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public class XYFLocalization extends XYLocalization {

	private long frame = 0;

	/**
	 * @return Get the frame number corresponding from this localization.
	 */
	public long getFrame() {
		return frame;
	}
	
	/**
	 * @param frame - frame
	 * @param x - x
	 * @param y - y
	 */
	public XYFLocalization(long frame, double x, double y) {
		super(x, y);

		this.frame = frame;
	}

	/**
	 * @param frame - frame
	 * @param x - x
	 * @param y - y
	 * @param ID - ID
	 */
	public XYFLocalization(long frame, double x, double y, long ID) {
		super(x, y, ID);

		this.frame = frame;
	}

}
