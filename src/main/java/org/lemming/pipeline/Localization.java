package org.lemming.pipeline;

import org.lemming.interfaces.LocalizationInterface;

/**
 * a localization implementation following the ViSP 2D format
 * 
 * @author Ronny Sczech
 *
 */
public class Localization implements LocalizationInterface {

	final private Number X;
	final private Number Y;
	private final long ID;
	static private long curID = 0;
	private boolean isLast;
	final private Long frame;
	final private Number intensity;
	
	public Localization(Number x, Number y, Number intensity, Long frame) {
		X=x; Y=y; ID=curID++; this.frame=frame; isLast=false; this.intensity = intensity;
	}
	
	public Localization(Long id, Number x, Number y, Number intensity, Long frame) {
		X=x; Y=y; ID=id; this.frame=frame; isLast=false; this.intensity = intensity;
	}
	
	
	public long getID() {
		return ID;
	}
	
	@Override
	public boolean isLast() {
		return isLast;
	}

	/**
	 * @param isLast - switch
	 */
	@Override
	public void setLast(boolean isLast) {
		this.isLast = isLast;
	}

	@Override
	public Number getX() {
		return X;
	}

	@Override
	public Number getY() {
		return Y;
	}

	@Override
	public Long getFrame() {
		return frame;
	}
	
	@Override
	public Number getIntensity() {
		return intensity;
	}

	@Override
	public String toString(){
		return "" + getID() + "\t" + getX() + "\t" + getY() + "\t" + getIntensity() + "\t" + getFrame();
	}	
}
