package org.lemming.pipeline;

public class XYZLocalization extends Localization{

	private double Z;

	public XYZLocalization(long frame, double x, double y, double z) {
		super(frame, x, y);
		this.Z=z;
	}
	
	public XYZLocalization(long ID, long frame, double x, double y, double z) {
		super(ID, frame, x, y);
		this.Z=z;
	}

	public double getZ(){
		return Z;
	}

}
