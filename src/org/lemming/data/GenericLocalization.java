package org.lemming.data;

/**
 * A localization with additional predefined and optional members.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public interface GenericLocalization extends Localization {
	/**
	 * Returns the appropriate member, or null if it does not exist.
	 * 
	 * Example : ROI r = g.get("ROI");
	 * 
	 * @param key
	 * @return
	 */
	public Object get(String member); 

	/**
	 * Checks if the specified member is present.
	 * 
	 * @param key
	 * @return
	 */
	public boolean has(String member);
	
	//////////// PREDEFINED COLS
	
	/**
	 * Returns the frame number or -1 if not available.
	 *  
	 * @return
	 */
	public long getFrame(); 

	/**
	 * Return the Z position or NaN if not available.
	 *  
	 * @return
	 */
	public double getZ(); 
	
	/**
	 * Returns the channel no. or -1 if not available.
	 * @return
	 */
	public int getChannel();
	
	public void setX(double x);
	public void setY(double y);
	public void setZ(double z);
	public void setFrame(long frame);
	public void setChannel(int channel);
	public void setID(long ID);
	
	public void set(String member, Object o);

}
