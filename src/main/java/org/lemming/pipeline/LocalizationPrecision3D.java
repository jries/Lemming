package org.lemming.pipeline;

public class LocalizationPrecision3D extends Localization{

	final private Number Z;
	final private Number sX;
	final private Number sY;
	final private Number sZ;

	public LocalizationPrecision3D(Number x, Number y, Number z, Number sx, Number sy, Number sz, Number intensity, Long frame) {
		super(x, y, intensity, frame);
		this.Z = z;
		this.sX = sx;
		this.sY = sy;
		this.sZ = sz;
	}

	public Number getZ(){
		return Z;
	}
	
	public Number getsX(){
		return sX;
	}
	
	public Number getsY(){
		return sY;
	}
	
	public Number getsZ(){
		return sZ;
	}
	
	@Override
	public String toString(){
		return "" + getX() + "\t" + getY() + "\t" + getZ() + "\t" + getsX() + "\t" + getsY() + "\t" + getsZ() + "\t" + getIntensity() +"\t" + getFrame();
	}
}
