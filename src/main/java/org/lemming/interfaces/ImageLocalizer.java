package org.lemming.interfaces;


/**
 * A localizer is transforms Frames into Localizations.
 * 
 * @author Thomas Pengo, Joe Borbely
 * @param <T> - data type
 * @param <F> - frame type
 */
public interface ImageLocalizer<T, F extends Frame<T>> extends Localizer, Well<F> {
}
