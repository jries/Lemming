package org.lemming.interfaces;


/**
 * A localization with additional predefined and optional members.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public interface GenericLocalization extends Localization {
	/**
	 * Example : ROI r = g.get("ROI");
	 * @param member - the appropriate member
	 * @return Returns the appropriate member, or null if it does not exist.
	 */
	public Object get(String member); 

	/**
	 * @param member - the appropriate member
	 * @return	Checks if the specified member is present.
	 */
	public boolean has(String member);
	
	//////////// PREDEFINED COLS
	
	/**
	 * @return Returns the frame number or -1 if not available.
	 */
	public long getFrame(); 

	/**
	 * @return Return the Z position or NaN if not available.
	 */
	public double getZ(); 
	
	/**
	 * @return Returns the channel no. or -1 if not available.
	 */
	public int getChannel();
	
	/**
	 * @param x - value
	 */
	public void setX(double x);
	/**
	 * @param y - value
	 */
	public void setY(double y);
	/**
	 * @param z - value
	 */
	public void setZ(double z);
	/**
	 * @param frame - the appropriate frame
	 */
	public void setFrame(long frame);
	/**
	 * @param channel - the appropriate channel
	 */
	public void setChannel(int channel);
	/**
	 * @param ID - value
	 */
	public void setID(long ID);
	
	/**
	 * @param member - the appropriate member
	 * @param o - object
	 */
	public void set(String member, Object o);

}
